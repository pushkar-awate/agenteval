"""The regression gate: has a candidate run gotten worse than the baseline?

This is selfheal-mlops' promotion guardrail, moved up a level: there, a retrained
model had to beat the current one on accuracy before it could be promoted; here,
a new prompt or model has to not *regress* the agent's eval metrics before it can
ship. Same shape - compare candidate vs a trusted baseline, block on a drop.

A metric is either *gated* (a drop past the tolerance fails the gate, so a worse
agent never ships) or *informational* (reported, never blocks - e.g. latency,
where "slower" is a trade-off, not a correctness regression). By default
task-success and every judge pass-rate are gated; latency and error count are
informational.
"""

# higher-is-better metrics that block a ship when they drop
_INFORMATIONAL = ("latency_p50_ms", "latency_p95_ms", "errors")


class Check:
    def __init__(self, metric, baseline, candidate, gated, tolerance=0.0, note=""):
        self.metric = metric
        self.baseline = baseline          # may be None (metric new in candidate)
        self.candidate = candidate
        self.gated = gated
        self.tolerance = tolerance
        self.note = note
        self.delta = None if baseline is None else (candidate - baseline)
        # a gated metric passes only if it stays within tolerance of the baseline
        self.passed = (not gated) or baseline is None or \
            candidate >= baseline - tolerance

    def to_dict(self):
        return {"metric": self.metric, "baseline": self.baseline,
                "candidate": self.candidate, "delta": self.delta,
                "gated": self.gated, "passed": self.passed, "note": self.note}


class GateResult:
    def __init__(self, checks):
        self.checks = checks

    @property
    def passed(self):
        return all(c.passed for c in self.checks)

    @property
    def regressions(self):
        return [c for c in self.checks if c.gated and not c.passed]

    def to_dict(self):
        return {"passed": self.passed, "regressions": [c.metric for c in self.regressions],
                "checks": [c.to_dict() for c in self.checks]}


def _judges(card):
    return card.get("per_judge_pass_rate", {}) or {}


def regression_gate(candidate, baseline, tolerance=0.0, gated=None):
    """Compare two scorecards (dicts from metrics.scorecard).

    tolerance: how many points a gated metric may drop before it fails (0 = any
    drop fails). gated: optional explicit set of gated metric names (e.g.
    {"task_success", "judge:exact_match"}); when None, the defaults apply.
    Returns a GateResult; a gated regression makes result.passed False.
    """
    def is_gated(metric, default):
        return default if gated is None else (metric in gated)

    checks = [Check("task_success", baseline.get("task_success"),
                    candidate.get("task_success", 0),
                    is_gated("task_success", True), tolerance)]

    bj, cj = _judges(baseline), _judges(candidate)
    for name in bj:                                   # judges present at baseline
        metric = "judge:%s" % name
        note = "judge absent in candidate run" if name not in cj else ""
        checks.append(Check(metric, bj.get(name), cj.get(name, 0),
                            is_gated(metric, True), tolerance, note=note))
    for name in cj:                                   # judges new in candidate
        if name not in bj:
            metric = "judge:%s" % name
            checks.append(Check(metric, None, cj.get(name), False, tolerance))

    for m in _INFORMATIONAL:                          # never blocks
        if m in baseline or m in candidate:
            checks.append(Check(m, baseline.get(m), candidate.get(m, 0), False, tolerance))

    return GateResult(checks)


def render_gate(result):
    lines = ["regression gate: %s" % ("PASS" if result.passed else "FAIL"),
             "-" * 56]
    for c in result.checks:
        tag = "gate" if c.gated else "info"
        base = " n/a " if c.baseline is None else ("%5.1f" % c.baseline)
        delta = "  -  " if c.delta is None else ("%+5.1f" % c.delta)
        flag = "" if c.passed else "  <-- REGRESSION"
        if c.note:
            flag += " (%s)" % c.note
        lines.append("  [%s] %-22s %s -> %5.1f  (%s)%s"
                     % (tag, c.metric, base, c.candidate, delta, flag))
    if result.regressions:
        lines.append("")
        lines.append("BLOCKED: gated regression in %s"
                     % ", ".join(c.metric for c in result.regressions))
    return "\n".join(lines)
