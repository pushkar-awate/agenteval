"""Aggregate a Run into a scorecard (task success, per-judge pass rates, latency).

Two subtleties, both there so a flaky or cached LLM judge never distorts the
numbers a gate trusts:

  * a judgement can be *not applicable* (an LLM judge whose call failed). It is
    neutral - excluded from task success and from that judge's pass-rate
    denominator - so an API hiccup reads as "un-evaluated", not as a regression.
  * token cost is counted only for REAL calls, not cache hits, so a fully-cached
    re-run reports $0 - which is the whole point of the cache.
"""


def _applicable(j):
    return getattr(j, "applicable", True)


def _meta(j):
    return getattr(j, "meta", None) or {}


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


def scorecard(run, price_per_1k_tokens=0.0):
    results = run.results
    n = len(results)

    # per-judge pass rate over APPLICABLE judgements only; a judge that never
    # applied (fully abstained) is dropped rather than shown as a fake 0%.
    judge_names = []
    for r in results:
        for j in r.judgements:
            if j.name not in judge_names:
                judge_names.append(j.name)
    per_judge = {}
    for name in judge_names:
        passed = sum(1 for r in results for j in r.judgements
                     if j.name == name and _applicable(j) and j.passed)
        total = sum(1 for r in results for j in r.judgements
                    if j.name == name and _applicable(j))
        if total:
            per_judge[name] = _pct(passed, total)

    # task success over EVALUATED cases (a crash counts as evaluated+failed; a
    # case whose every judge abstained is un-evaluated, not a failure).
    evaluated = [r for r in results if r.evaluated]
    lat = [r.latency_ms for r in results]

    # cost/calls: a cache hit is not a billed call.
    est_tokens = sum(_meta(j).get("est_tokens", 0)
                     for r in results for j in r.judgements if not _meta(j).get("cached"))
    llm_calls = sum(1 for r in results for j in r.judgements
                    if "est_tokens" in _meta(j) and not _meta(j).get("cached"))
    cache_hits = sum(1 for r in results for j in r.judgements if _meta(j).get("cached"))

    return {
        "version": run.version,
        "n_cases": n,
        "evaluated_cases": len(evaluated),
        "unevaluated_cases": n - len(evaluated),
        "task_success": _pct(sum(1 for r in evaluated if r.passed), len(evaluated)),
        "per_judge_pass_rate": per_judge,
        "errors": sum(1 for r in results if r.error),
        "latency_p50_ms": round(_percentile(lat, 50), 2),
        "latency_p95_ms": round(_percentile(lat, 95), 2),
        "est_tokens": est_tokens,
        "est_cost_usd": round(est_tokens / 1000.0 * price_per_1k_tokens, 6),
        "llm_calls": llm_calls,
        "llm_cache_hits": cache_hits,
    }


def render_scorecard(card):
    head = "task success : %d%%  (%d cases" % (card["task_success"], card["n_cases"])
    if card.get("unevaluated_cases"):
        head += ", %d un-evaluated" % card["unevaluated_cases"]
    head += ")"
    lines = ["scorecard: %s" % card["version"], "-" * 40, head]
    for name, rate in card["per_judge_pass_rate"].items():
        lines.append("  %-14s: %d%%" % (name, rate))
    lines += ["errors       : %d" % card["errors"],
              "latency p50  : %.1f ms" % card["latency_p50_ms"],
              "latency p95  : %.1f ms" % card["latency_p95_ms"]]
    if card.get("llm_calls") or card.get("llm_cache_hits"):
        lines.append("llm calls    : %d  (cache hits %d)"
                     % (card.get("llm_calls", 0), card.get("llm_cache_hits", 0)))
        lines.append("est. cost    : $%.4f  (~%d tokens, billed calls only)"
                     % (card.get("est_cost_usd", 0.0), card.get("est_tokens", 0)))
    return "\n".join(lines)
