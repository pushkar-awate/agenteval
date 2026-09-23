"""agenteval dashboard - run an eval, see the scorecard, and get the regression
verdict against the committed baseline.

This is the human view of the same engine the CLI and CI use: nothing here
computes metrics itself, it calls evalcore. Deploys on Streamlit Cloud with only
`streamlit` as a dependency (the core is standard library).

    streamlit run streamlit_app.py
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evalcore.dataset import load_cases
from evalcore.runner import run_eval
from evalcore.metrics import scorecard
from evalcore.registry import Registry
from evalcore.gate import regression_gate
from evalcore.run import build_judges, _degraded, ROOT
from app.demo_target import classify

st.set_page_config(page_title="agenteval", page_icon="✅", layout="wide")

CASES = os.path.join(ROOT, "app", "cases", "demo_cases.jsonl")
BASELINE = os.path.join(ROOT, "baseline.json")


@st.cache_data
def _cases():
    return load_cases(CASES)


st.title("agenteval")
st.caption("Eval + regression gate for LLM agents - *self-healing MLOps, but for agents.* "
           "Store a good run as a baseline; block any change that makes the agent worse.")

# --- controls --------------------------------------------------------------
with st.sidebar:
    st.header("Run")
    agent = st.radio("System-under-test",
                     ["healthy agent", "degraded agent (broke one intent)"],
                     help="The degraded agent simulates a prompt change that "
                          "stopped recognising 'technical' issues - use it to "
                          "watch the gate catch a regression.")
    has_key = bool(os.environ.get("GROQ_API_KEY"))
    use_llm = st.toggle("LLM-as-judge (faithfulness + relevance)", value=False,
                        disabled=not has_key,
                        help="Needs GROQ_API_KEY. Off by default so the demo is "
                             "deterministic and free.")
    if not has_key:
        st.caption("Set GROQ_API_KEY to enable the LLM judges.")
    tolerance = st.slider("Gate tolerance (points a gated metric may drop)",
                          0.0, 20.0, 0.0, 1.0)
    run_it = st.button("Run eval", type="primary", use_container_width=True)

reg = Registry(root=os.path.join(ROOT, "runs"), baseline_path=BASELINE)
baseline = reg.baseline_scorecard()

if not run_it:
    st.info("Pick a system-under-test in the sidebar and press **Run eval**.")
    if baseline:
        st.subheader("Current baseline")
        st.caption("Version `%s` - new runs are gated against this."
                   % baseline.get("version", "?"))
        st.json({k: baseline[k] for k in ("task_success", "per_judge_pass_rate",
                                          "n_cases") if k in baseline})
    st.stop()

# --- run -------------------------------------------------------------------
target = _degraded if agent.startswith("degraded") else classify
judges = build_judges(use_llm=use_llm, cache_dir=os.path.join(ROOT, "runs"))
run = run_eval(target, _cases(), judges, version="dashboard")
card = scorecard(run)

# --- scorecard -------------------------------------------------------------
st.subheader("Scorecard")
cols = st.columns(4)
cols[0].metric("Task success", "%d%%" % card["task_success"])
cols[1].metric("Cases", card["n_cases"])
cols[2].metric("Errors", card["errors"])
cols[3].metric("Latency p95", "%.1f ms" % card["latency_p95_ms"])
jcols = st.columns(max(1, len(card["per_judge_pass_rate"])))
for col, (name, rate) in zip(jcols, card["per_judge_pass_rate"].items()):
    col.metric(name, "%d%%" % rate)
if card.get("llm_calls"):
    st.caption("LLM judge: %d calls, %d cache hits, ~%d tokens"
               % (card["llm_calls"], card["llm_cache_hits"], card["est_tokens"]))

# --- regression verdict ----------------------------------------------------
st.subheader("Regression gate")
if not baseline:
    st.warning("No baseline committed yet - run `python -m evalcore.run "
               "--set-baseline` to create one.")
else:
    result = regression_gate(card, baseline, tolerance=tolerance)
    if result.passed:
        st.success("PASS - no gated regression vs baseline `%s`."
                   % baseline.get("version", "?"))
    else:
        st.error("FAIL - gated regression in %s. This change would be blocked."
                 % ", ".join(c.metric for c in result.regressions))

    # eval-drift: baseline vs candidate on each gated metric
    rows = {"metric": [], "baseline": [], "candidate": []}
    for c in result.checks:
        if c.gated:
            rows["metric"].append(c.metric)
            rows["baseline"].append(c.baseline if c.baseline is not None else 0)
            rows["candidate"].append(c.candidate)
    if rows["metric"]:
        try:
            import pandas as pd
            df = pd.DataFrame(rows).set_index("metric")
            st.bar_chart(df)
        except Exception:
            st.table(rows)
    st.dataframe([c.to_dict() for c in result.checks], use_container_width=True)

# --- trace viewer ----------------------------------------------------------
st.subheader("Traces")
st.caption("Every case, its output, and each judge's verdict.")
for r in run.results:
    ok = "PASS" if r.passed else "FAIL"
    with st.expander("%s  -  %s" % (ok, str(r.case.input)[:80])):
        st.write("**Expected:** ", r.case.expected)
        st.write("**Output:** ", r.output or "_(empty)_")
        if r.error:
            st.error("error: %s" % r.error)
        st.table([{"judge": j.name, "score": round(j.score, 2),
                   "passed": j.passed, "detail": j.detail}
                  for j in r.judgements])
