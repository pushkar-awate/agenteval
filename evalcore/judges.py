"""Judges score one (case, output) pair.

A judge returns a Judgement(name, score in [0,1], passed, detail). The judges
here are the deterministic "mock" family: rule-based, need no API key, and make
tests reproducible - mirroring agentcore's MockBrain. An LLM-as-judge that reuses
agentcore's Brain interface is added in a later milestone.
"""
import re


class Judgement:
    def __init__(self, name, score, passed, detail=""):
        self.name = name
        self.score = float(score)
        self.passed = bool(passed)
        self.detail = detail

    def to_dict(self):
        return {"name": self.name, "score": self.score,
                "passed": self.passed, "detail": self.detail}


class Judge:
    name = "judge"

    def score(self, case, output):
        raise NotImplementedError


def _norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())


class ExactMatch(Judge):
    name = "exact_match"

    def score(self, case, output):
        ok = _norm(output) == _norm(case.expected)
        return Judgement(self.name, 1.0 if ok else 0.0, ok,
                         "" if ok else "expected %r, got %r" % (case.expected, output))


class Contains(Judge):
    name = "contains"

    def score(self, case, output):
        ok = _norm(case.expected) in _norm(output)
        return Judgement(self.name, 1.0 if ok else 0.0, ok)


class Regex(Judge):
    name = "regex"

    def __init__(self, pattern, name=None):
        self.pattern = re.compile(pattern)
        if name:
            self.name = name

    def score(self, case, output):
        ok = bool(self.pattern.search(str(output)))
        return Judgement(self.name, 1.0 if ok else 0.0, ok)
