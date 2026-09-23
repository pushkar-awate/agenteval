"""Aggregate a Run into a scorecard (task success, per-judge pass rates, latency)."""


def _pct(n, d):
    return round(100.0 * n / d) if d else 0


def _percentile(values, p):
    if not values:
        return 0.0
    xs = sorted(values)
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def scorecard(run):
    results = run.results
    n = len(results)
    judge_names = []
    for r in results:
        for j in r.judgements:
            if j.name not in judge_names:
                judge_names.append(j.name)
    per_judge = {}
    for name in judge_names:
        passed = sum(1 for r in results for j in r.judgements
                     if j.name == name and j.passed)
        total = sum(1 for r in results for j in r.judgements if j.name == name)
        per_judge[name] = _pct(passed, total)
    lat = [r.latency_ms for r in results]
    return {
        "version": run.version,
        "n_cases": n,
        "task_success": _pct(sum(1 for r in results if r.passed), n),
        "per_judge_pass_rate": per_judge,
        "errors": sum(1 for r in results if r.error),
        "latency_p50_ms": round(_percentile(lat, 50), 2),
        "latency_p95_ms": round(_percentile(lat, 95), 2),
    }


def render_scorecard(card):
    lines = ["scorecard: %s" % card["version"], "-" * 40,
             "task success : %d%%  (%d cases)" % (card["task_success"], card["n_cases"])]
    for name, rate in card["per_judge_pass_rate"].items():
        lines.append("  %-14s: %d%%" % (name, rate))
    lines += ["errors       : %d" % card["errors"],
              "latency p50  : %.1f ms" % card["latency_p50_ms"],
              "latency p95  : %.1f ms" % card["latency_p95_ms"]]
    return "\n".join(lines)
