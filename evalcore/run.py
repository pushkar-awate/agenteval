"""CLI for running the demo eval, storing runs, and gating regressions.

  python -m evalcore.run                     # run the demo eval, print scorecard
  python -m evalcore.run --save              # also snapshot the run under runs/
  python -m evalcore.run --set-baseline      # store this run as the baseline
  python -m evalcore.run --gate              # compare vs baseline, fail on regression
  python -m evalcore.run --degrade --gate    # demo: a broken agent, gate blocks it

--gate exits non-zero on a gated regression, so it drops straight into CI (M4).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evalcore.dataset import load_cases
from evalcore.judges import ExactMatch
from evalcore.runner import run_eval
from evalcore.metrics import scorecard, render_scorecard
from evalcore.registry import Registry
from evalcore.gate import regression_gate, render_gate
from app.demo_target import classify

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _degraded(text):
    """A deliberately broken agent: a 'prompt change' that stopped recognising
    technical issues. Used to demonstrate the gate catching a regression."""
    label = classify(text)
    return "other" if label == "technical" else label


def main(argv=None):
    ap = argparse.ArgumentParser(description="agenteval - run, store and gate agent evals")
    ap.add_argument("--version", default="demo-v1", help="label for this run")
    ap.add_argument("--save", action="store_true", help="snapshot the run under runs/")
    ap.add_argument("--set-baseline", action="store_true", dest="set_baseline",
                    help="store this run as the regression baseline")
    ap.add_argument("--gate", action="store_true",
                    help="compare against the baseline and fail on a gated regression")
    ap.add_argument("--tolerance", type=float, default=0.0,
                    help="points a gated metric may drop before the gate fails")
    ap.add_argument("--degrade", action="store_true",
                    help="run a deliberately worse agent (to demo the gate)")
    ap.add_argument("--registry", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--baseline-path", default=os.path.join(ROOT, "baseline.json"))
    args = ap.parse_args(argv)

    cases = load_cases(os.path.join(ROOT, "app", "cases", "demo_cases.jsonl"))
    target = _degraded if args.degrade else classify
    run = run_eval(target, cases, [ExactMatch()], version=args.version)
    card = scorecard(run)
    print(render_scorecard(card))

    reg = Registry(root=args.registry, baseline_path=args.baseline_path)
    if args.save:
        path = reg.save_run(run, card)
        print("\nsaved run -> %s" % os.path.relpath(path, ROOT))
    if args.set_baseline:
        path = reg.set_baseline(run, card)
        print("\nbaseline set -> %s  (version %s)" % (os.path.relpath(path, ROOT), run.version))

    if args.gate:
        base = reg.baseline_scorecard()
        if base is None:
            print("\nno baseline found at %s - run --set-baseline first"
                  % os.path.relpath(args.baseline_path, ROOT), file=sys.stderr)
            return 2
        result = regression_gate(card, base, tolerance=args.tolerance)
        print("\n" + render_gate(result))
        return 0 if result.passed else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
