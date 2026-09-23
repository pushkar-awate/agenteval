# agenteval

[![eval](https://github.com/pushkar-awate/agenteval/actions/workflows/eval.yml/badge.svg)](https://github.com/pushkar-awate/agenteval/actions/workflows/eval.yml)

**Try it live: https://agenteval-pushkar.streamlit.app**

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
3. **Judges** - score each `(case, output)`. A deterministic family needs no
   key: `ExactMatch`, `Contains`, `Regex`, `FormatAdherence` and a
   `GuardrailJudge` (guardrail pass rate). An **LLM-as-judge** (`--llm`) adds
   faithfulness and relevance, reusing agentcore's Brain and caching every
   answer so re-runs don't re-bill; an embedding judge comes next.
4. **Metrics** - aggregate into a scorecard: task success, per-judge pass rate
   (incl. format adherence and guardrail pass rate), latency p50/p95, and an
   estimated token cost when an LLM judge runs.

The **brain is task-agnostic**: a pluggable reasoning backend that is deterministic and key-free by default, and swaps to a real hosted LLM via `complete(prompt)` - which is what the LLM-as-judge calls to score an answer. No agent- or app-specific logic lives in the shared core.

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

## Dashboard

A Streamlit dashboard ([live](https://agenteval-pushkar.streamlit.app)) is the
human view of the same engine the CLI and CI use:
pick a system-under-test, run the eval, and see the scorecard, a
baseline-vs-candidate eval-drift chart, the pass/fail regression verdict, and a
per-case trace viewer (input, output, every judge). Toggle the degraded agent to
watch the gate catch a regression live.

```
streamlit run streamlit_app.py
```

It deploys on Streamlit Cloud with only `streamlit` as a dependency (the core is
standard library); set `GROQ_API_KEY` in the app secrets to enable the
LLM-as-judge.

## Self-hosted judge & infra (optional)

Two extras that round out the story:

- **`notebooks/lora_judge_colab.ipynb`** trains a small LoRA adapter as a
  faithfulness judge on a free Colab GPU, then loads it CPU-side - a self-hosted
  judge that plugs into the same `Judge` interface as the deterministic and
  hosted-LLM judges, with no per-call cost.
- **`infra/`** is a Terraform module provisioning the cloud backing for eval
  history (a versioned S3 bucket for baselines/runs + a DynamoDB run index). It
  runs free and offline against LocalStack, so `terraform apply` needs no AWS
  account.

## Project layout

```
agentcore/     reusable runtime shared with jobfit-agent and selfheal-mlops
evalcore/      the eval engine: dataset, judges, runner, metrics,
               registry (versioned runs + baseline) and gate (regression guard)
app/           the system-under-test being evaluated + its labelled cases
streamlit_app.py  the dashboard (run eval, drift chart, verdict, traces)
baseline.json  the committed baseline the regression gate scores runs against
infra/         Terraform module (S3 + DynamoDB) for hosted run storage, LocalStack-ready
notebooks/     a LoRA-trained, CPU-served faithfulness judge (free Colab GPU)
tests/         runnable with plain python (no pytest)
SPEC.md        full spec and build checklist
```

The regression gate (M1), LLM-as-judge + format/guardrail metrics + caching
(M2), Streamlit dashboard (M3), CI eval gate (M4), and the self-hosted LoRA
judge + Terraform/LocalStack infra module (M5) are all in - every push runs the
tests and gates the agent against the committed baseline. The core is pure
standard library; the LLM judge is opt-in via `--llm` and a free `GROQ_API_KEY`.
