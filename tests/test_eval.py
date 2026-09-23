"""Runnable with plain `python tests/test_eval.py` (no pytest needed)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from evalcore.dataset import EvalCase, load_cases
from evalcore.judges import ExactMatch, Contains, Regex
from evalcore.runner import run_eval
from evalcore.metrics import scorecard
from evalcore.registry import Registry
from evalcore.gate import regression_gate
from evalcore.judges import (LLMJudge, FormatAdherence, GuardrailJudge,
                             faithfulness_judge)
from evalcore.cache import JSONLCache
from agentcore.brain import MockBrain
from app.demo_target import classify


def _cases():
    return load_cases(os.path.join(ROOT, "app", "cases", "demo_cases.jsonl"))


def test_judges():
    c = EvalCase("1", "x", expected="billing")
    assert ExactMatch().score(c, "billing").passed
    assert not ExactMatch().score(c, "technical").passed
    assert Contains().score(c, "this is a billing issue").passed
    assert Regex(r"^\d{3}$").score(c, "500").passed
    print("PASS: judges  (exact-match, contains, regex)")


def test_runner_and_scorecard():
    cases = _cases()
    card = scorecard(run_eval(classify, cases, [ExactMatch()], version="test"))
    assert card["n_cases"] == len(cases)
    assert card["task_success"] >= 80, "demo classifier should score >=80%%, got %d" % card["task_success"]
    assert card["errors"] == 0
    assert card["latency_p50_ms"] >= 0
    print("PASS: runner + scorecard  (task_success=%d%%, n=%d)"
          % (card["task_success"], card["n_cases"]))


def test_worse_target_scores_lower():
    cases = _cases()
    good = scorecard(run_eval(classify, cases, [ExactMatch()]))["task_success"]
    bad = scorecard(run_eval(lambda x: "other", cases, [ExactMatch()]))["task_success"]
    assert bad < good, "a worse system-under-test must score lower"
    print("PASS: worse system-under-test scores lower  (%d%% < %d%%)" % (bad, good))


def test_crashing_target_is_a_failed_case_not_a_crash():
    def boom(x):
        raise ValueError("kaboom")
    cases = _cases()[:3]
    card = scorecard(run_eval(boom, cases, [ExactMatch()]))
    assert card["errors"] == len(cases) and card["task_success"] == 0, \
        "a crashing SUT should be recorded as failed cases, not raise"
    print("PASS: crashing system-under-test recorded as failed cases")


def test_registry_roundtrip():
    import tempfile, shutil
    tmp = tempfile.mkdtemp()
    try:
        reg = Registry(root=os.path.join(tmp, "runs"),
                       baseline_path=os.path.join(tmp, "baseline.json"))
        run = run_eval(classify, _cases(), [ExactMatch()], version="v1")
        reg.save_run(run)
        assert "v1" in reg.list_runs()
        loaded = reg.load_run("v1")
        assert loaded["version"] == "v1" and "scorecard" in loaded and "run" in loaded
        assert reg.load_baseline() is None, "no baseline until one is set"
        reg.set_baseline(run)
        base = reg.baseline_scorecard()
        assert base["task_success"] == scorecard(run)["task_success"]
        print("PASS: registry roundtrip  (save/load run, set/load baseline)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_gate_passes_when_not_worse():
    cases = _cases()
    base = scorecard(run_eval(classify, cases, [ExactMatch()], version="base"))
    same = scorecard(run_eval(classify, cases, [ExactMatch()], version="cand"))
    res = regression_gate(same, base)
    assert res.passed and not res.regressions, "identical run must pass the gate"
    # a strictly better candidate also passes
    better = dict(same, task_success=min(100, same["task_success"] + 5))
    assert regression_gate(better, base).passed
    print("PASS: gate passes when candidate is not worse")


def test_gate_blocks_regression():
    from evalcore.run import _degraded
    cases = _cases()
    base = scorecard(run_eval(classify, cases, [ExactMatch()], version="base"))
    bad = scorecard(run_eval(_degraded, cases, [ExactMatch()], version="bad"))
    res = regression_gate(bad, base)
    assert not res.passed, "a degraded agent must fail the gate"
    assert "task_success" in [c.metric for c in res.regressions]
    assert bad["task_success"] < base["task_success"]
    print("PASS: gate blocks regression  (task_success %d%% < %d%%)"
          % (bad["task_success"], base["task_success"]))


def test_gate_tolerance():
    base = {"task_success": 90, "per_judge_pass_rate": {"exact_match": 90}}
    cand = {"task_success": 85, "per_judge_pass_rate": {"exact_match": 85}}
    assert not regression_gate(cand, base, tolerance=0.0).passed, "5pt drop, zero tol -> fail"
    assert regression_gate(cand, base, tolerance=5.0).passed, "5pt drop within 5pt tol -> pass"
    print("PASS: gate tolerance  (5pt drop fails at tol=0, passes at tol=5)")


class _CountingStubBrain(MockBrain):
    """A no-network LLM stand-in: returns a fixed JSON judgement and counts how
    many times complete() actually runs (to prove caching skips calls)."""
    def __init__(self, score=90):
        super().__init__()
        self.calls = 0
        self._score = score
    def complete(self, prompt):
        self.calls += 1
        return '```json\n{"score": %d, "reason": "looks faithful"}\n```' % self._score


def test_llm_judge_with_stub_and_cache():
    import tempfile, os as _os
    c = EvalCase("1", "Why was I charged twice?", expected="billing")
    tmp = tempfile.mkdtemp()
    try:
        cache = JSONLCache(path=_os.path.join(tmp, "judge_cache.jsonl"))
        brain = _CountingStubBrain(score=90)
        judge = faithfulness_judge(brain, cache=cache)
        j1 = judge.score(c, "This is a billing problem.")
        assert j1.passed and abs(j1.score - 0.9) < 1e-6, "stub score 90 -> 0.9, pass"
        assert j1.meta.get("est_tokens", 0) > 0
        assert brain.calls == 1
        # second identical score must hit the cache, not the brain
        j2 = judge.score(c, "This is a billing problem.")
        assert j2.passed and brain.calls == 1, "cache should prevent a second call"
        assert j2.meta.get("cached") is True
        # a fresh cache object reads the persisted file
        assert len(JSONLCache(path=cache.path)) == 1
        print("PASS: LLM judge via stub + response caching (1 call, cache hit on repeat)")
    finally:
        import shutil; shutil.rmtree(tmp, ignore_errors=True)


def test_llm_judge_abstains_on_mock_brain():
    c = EvalCase("1", "x", expected="billing")
    j = faithfulness_judge(MockBrain()).score(c, "anything")
    assert (not j.passed) and j.meta.get("abstained"), "MockBrain -> judge abstains"
    print("PASS: LLM judge abstains on the key-free MockBrain")


def test_format_and_guardrail_judges():
    labels = FormatAdherence(r"billing|technical|account|shipping|other", name="format")
    c = EvalCase("1", "x", expected="billing")
    assert labels.score(c, "billing").passed
    assert not labels.score(c, "totally invalid label").passed
    g = GuardrailJudge()
    assert g.score(c, "billing").passed
    assert not g.score(c, "").passed, "empty output fails the guardrail"
    assert not g.score(c, "As an AI language model, I cannot help").passed
    print("PASS: format-adherence + guardrail judges")


def test_cost_estimate_in_scorecard():
    cases = _cases()[:3]
    brain = _CountingStubBrain(score=80)
    run = run_eval(classify, cases, [ExactMatch(), faithfulness_judge(brain)])
    card = scorecard(run, price_per_1k_tokens=0.5)
    assert card["llm_calls"] == len(cases), "one LLM-judge call per case"
    assert card["est_tokens"] > 0 and card["est_cost_usd"] > 0
    assert "faithfulness" in card["per_judge_pass_rate"]
    print("PASS: cost estimate  (%d llm calls, ~%d tok, $%.4f)"
          % (card["llm_calls"], card["est_tokens"], card["est_cost_usd"]))


if __name__ == "__main__":
    test_judges()
    test_runner_and_scorecard()
    test_worse_target_scores_lower()
    test_crashing_target_is_a_failed_case_not_a_crash()
    test_registry_roundtrip()
    test_gate_passes_when_not_worse()
    test_gate_blocks_regression()
    test_gate_tolerance()
    test_llm_judge_with_stub_and_cache()
    test_llm_judge_abstains_on_mock_brain()
    test_format_and_guardrail_judges()
    test_cost_estimate_in_scorecard()
    print("\nAll tests passed.")
