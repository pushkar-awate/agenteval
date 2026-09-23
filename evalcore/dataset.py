"""Eval cases: the inputs a system-under-test is scored against (JSONL on disk)."""
import json


class EvalCase:
    def __init__(self, id, input, expected=None, tags=None, meta=None):
        self.id = id
        self.input = input
        self.expected = expected
        self.tags = tags or []
        self.meta = meta or {}

    @classmethod
    def from_dict(cls, d):
        return cls(d.get("id"), d["input"], d.get("expected"),
                   d.get("tags"), d.get("meta"))

    def to_dict(self):
        return {"id": self.id, "input": self.input, "expected": self.expected,
                "tags": self.tags, "meta": self.meta}


def load_cases(path):
    cases = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                cases.append(EvalCase.from_dict(json.loads(line)))
    return cases


def save_cases(path, cases):
    with open(path, "w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(json.dumps(c.to_dict()) + "\n")
