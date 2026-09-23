# agenteval

An eval + observability harness for LLM agents. It answers the question every
team shipping agents keeps asking: **"did that prompt or model change make my
agent better, or did I just quietly break it?"**

You point it at a system-under-test and a set of eval cases, and it runs the
agent, scores each case with pluggable judges, and produces a scorecard. Store a
run as a baseline and later runs are compared against it: if a change makes the
agent worse on a gated metric, the **regression gate** fails - so a worse agent
never ships. Think of it as *self-healing MLOps, but for LLM agents*: the same
drift-and-guardrail idea as
[selfheal-mlops](https://github.com/pushkar-awate/selfheal-mlops), applied to
prompts and models instead of a tabular model.

It runs with **zero dependencies and zero API keys** by default: the judges are
deterministic, so a fresh clone scores an example agent immediately.

```
$ python -m evalcore.run
scorecard: demo-v1
----------------------------------------
task success : 100%  (12 cases)
  exact_match   : 100%
errors       : 0
latency p50  : 0.0 ms
latency p95  : 0.0 ms
```

## How it works

`evalcore/` is a small, task-agnostic engine:

1. **Dataset** - eval cases (input, expected/rubric) as JSONL.
2. **Runner** - runs any `target(input) -> output` over the cases, times each
   call, and records a crashing target as a failed case (never a crash).
3. **Judges** - score each `(case, output)`: `ExactMatch`, `Contains`, `Regex`
   today (deterministic, no key); an LLM-as-judge and embedding judge next.
4. **Metrics** - aggregate into a scorecard: task success, per-judge pass rate,
   latency p50/p95.

The **brain is task-agnostic**: a pluggable reasoning backend that is deterministic and key-free by default, and swaps to a real hosted LLM via `complete(prompt)` - which is what the LLM-as-judge will call to score an answer. No agent- or app-specific logic lives in the shared core.

The engine knows nothing about the agent it scores - swap the `target` and the
same runner, judges and metrics evaluate a different system. It is built on the
reusable [`agentcore`](https://github.com/pushkar-awate/jobfit-agent) runtime,
the third app on that core.

## Regression gate - the wedge

Scoring an agent once is easy; catching the day a prompt or model change quietly
makes it *worse* is the hard part. agenteval stores a run as a **baseline** and
gates every later run against it: if a gated metric drops, the gate fails, so a
worse agent never ships. This is exactly selfheal-mlops' promotion guardrail,
lifted from a tabular model up to an LLM agent.

```
# store a known-good run as the baseline (a single committed file)
python -m evalcore.run --set-baseline

# later, in CI: run again and gate against that baseline
python -m evalcore.run --gate          # exits non-zero on a gated regression
```

A healthy change passes; a broken one is blocked, with the offending metrics named:

```
$ python -m evalcore.run --degrade --gate   # a "prompt change" that broke one intent
regression gate: FAIL
  [gate] task_success       100.0 ->  75.0  (-25.0)  <-- REGRESSION
  [gate] judge:exact_match  100.0 ->  75.0  (-25.0)  <-- REGRESSION
BLOCKED: gated regression in task_success, judge:exact_match
```

Every metric is either **gated** (a drop fails the gate - task success, judge
pass rates) or **informational** (reported, never blocks - latency, error
count). `--tolerance` sets how many points a gated metric may slip before it
fails. The baseline is one committed `baseline.json` (scorecard plus full
traces), so CI can gate a pull request with just that file - no database.

## Project layout

```
agentcore/     reusable runtime shared with jobfit-agent and selfheal-mlops
evalcore/      the eval engine: dataset, judges, runner, metrics,
               registry (versioned runs + baseline) and gate (regression guard)
app/           the system-under-test being evaluated + its labelled cases
baseline.json  the committed baseline the regression gate scores runs against
tests/         runnable with plain python (no pytest)
SPEC.md        full spec and build checklist
```

The registry and regression gate are in (M1). See `SPEC.md` for what's next:
LLM-as-judge (faithfulness/relevance), a Streamlit dashboard, and a GitHub
Actions eval gate. Core is pure standard library; Python 3.8+.
