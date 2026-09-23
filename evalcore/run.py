"""CLI: `python -m evalcore.run` runs the demo eval and prints a scorecard."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evalcore.dataset import load_cases
from evalcore.judges import ExactMatch
from evalcore.runner import run_eval
from evalcore.metrics import scorecard, render_scorecard
from app.demo_target import classify


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cases = load_cases(os.path.join(root, "app", "cases", "demo_cases.jsonl"))
    run = run_eval(classify, cases, [ExactMatch()], version="demo-v1")
    print(render_scorecard(scorecard(run)))


if __name__ == "__main__":
    main()
