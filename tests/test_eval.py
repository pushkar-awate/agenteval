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


if __name__ == "__main__":
    test_judges()
    test_runner_and_scorecard()
    test_worse_target_scores_lower()
    test_crashing_target_is_a_failed_case_not_a_crash()
    test_registry_roundtrip()
    test_gate_passes_when_not_worse()
    test_gate_blocks_regression()
    test_gate_tolerance()
    print("\nAll tests passed.")
