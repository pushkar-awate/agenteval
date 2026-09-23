"""Versioned run storage + a committed baseline, for regression gating.

Two things live here:

  * a run *history* under ``runs/`` - one JSON snapshot per run (scorecard +
    full traces). This is local, gitignored, and is what a dashboard reads.
  * a single *baseline* file (``baseline.json``, committed) that the regression
    gate scores new runs against. Keeping the baseline self-contained means CI
    can gate a pull request with only that one file checked out - no history,
    no database.

Pure standard library, like the rest of the core.
"""
import json
import os
import time

from .metrics import scorecard as _scorecard


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe(version):
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(version))


def _envelope(run, card):
    """A self-describing record: metrics for gating, traces for inspection."""
    return {"version": run.version, "saved_at": _now(),
            "scorecard": card, "run": run.to_dict()}


class Registry:
    def __init__(self, root="runs", baseline_path="baseline.json"):
        self.root = root
        self.baseline_path = baseline_path

    # --- run history -------------------------------------------------------
    def save_run(self, run, card=None):
        card = card or _scorecard(run)
        if not os.path.isdir(self.root):
            os.makedirs(self.root, exist_ok=True)
        path = os.path.join(self.root, _safe(run.version) + ".json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(_envelope(run, card), fh, indent=2)
        return path

    def load_run(self, version):
        path = os.path.join(self.root, _safe(version) + ".json")
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def list_runs(self):
        if not os.path.isdir(self.root):
            return []
        return sorted(f[:-5] for f in os.listdir(self.root) if f.endswith(".json"))

    # --- baseline ----------------------------------------------------------
    def set_baseline(self, run, card=None):
        card = card or _scorecard(run)
        with open(self.baseline_path, "w", encoding="utf-8") as fh:
            json.dump(_envelope(run, card), fh, indent=2)
        return self.baseline_path

    def load_baseline(self):
        if not os.path.exists(self.baseline_path):
            return None
        with open(self.baseline_path, encoding="utf-8") as fh:
            return json.load(fh)

    def baseline_scorecard(self):
        b = self.load_baseline()
        return b.get("scorecard") if b else None
