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


if __name__ == "__main__":
    test_judges()
    test_runner_and_scorecard()
    test_worse_target_scores_lower()
    test_crashing_target_is_a_failed_case_not_a_crash()
    print("\nAll tests passed.")
