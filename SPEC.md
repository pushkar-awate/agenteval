# agenteval - spec & build checklist

An eval + observability harness for LLM agents. "Self-healing MLOps, but for
agents": detect when a prompt/model change makes an agent worse, and block the
regression before it ships. Built on the same `agentcore` runtime as
[jobfit-agent](https://github.com/pushkar-awate/jobfit-agent) and
[selfheal-mlops](https://github.com/pushkar-awate/selfheal-mlops).

## The wedge
selfheal-mlops detects *data drift* and gates a model promotion. agenteval
detects *eval drift* for an agent and gates a prompt/model change the same way.
Same mental model, applied to LLM systems.

## Architecture
- `agentcore/`  - reused runtime (loop, brain, tools, memory, guardrails).
- `evalcore/`   - task-agnostic eval engine:
  - `dataset.py`  - EvalCase + JSONL load/save.
  - `judges.py`   - Judge base; ExactMatch/Contains/Regex (deterministic "mock"
    family, no key). LLMJudge (reuses agentcore Brain) + EmbedJudge come later.
  - `runner.py`   - run_eval(target, cases, judges) -> Run; times each call and
    records a crashing target as a failed case, not a crash.
  - `metrics.py`  - scorecard: task success, per-judge pass rate, latency p50/p95.
  - `registry.py` - versioned runs + a baseline (M1).
  - `gate.py`     - regression gate vs baseline (M1) - the guardrail.
- `app/`        - the system-under-test + its eval task (demo intent classifier
  now; jobfit-agent's assess_fit next).
- `streamlit_app.py` - dashboard: run eval, metric table, eval-drift chart,
  regression verdict, trace viewer (M3).
- `.github/workflows/eval.yml` - CI eval + regression gate (M4).
- `infra/`      - Terraform module against LocalStack (M5).

## Metrics
task success | faithfulness (LLM judge) | answer relevance (LLM judge) |
format adherence | guardrail pass rate | latency p50/p95 | est. cost | stability
(variance across N repeats). Each metric can be *gated* (drop fails CI) or
*informational*.

## Milestones
- [x] M0 - scaffold: evalcore (dataset/judges/runner/metrics), demo
  system-under-test + labelled cases, CLI, green tests.
- [ ] M1 - registry + regression gate (the eval-drift wedge).
- [ ] M2 - LLMJudge (faithfulness/relevance) + guardrail/format metrics +
  latency/cost + response caching.
- [ ] M3 - Streamlit dashboard (metric table, eval-drift chart, verdict, traces).
- [ ] M4 - GitHub Actions eval + regression gate + badge.
- [ ] M5 - PyTorch LoRA judge (Colab) + Terraform/LocalStack infra module.
- [ ] M6 - README with the "eval drift for agents" framing, demo GIF, limits.

## Free-to-build notes / traps
- MockJudge default -> zero keys, reproducible tests.
- LLMJudge caps calls per case and caches to JSONL (avoid re-billing / 429s).
- Embeddings: MiniLM on CPU; keep eval sets small for Streamlit RAM.
- LoRA training on free Colab GPU, never on Streamlit; load adapter CPU-side.
