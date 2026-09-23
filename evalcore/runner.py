"""Run a system-under-test over eval cases and score each with judges."""
import time

from .judges import Judgement


class CaseResult:
    def __init__(self, case, output, latency_ms, judgements, error=None):
        self.case = case
        self.output = output
        self.latency_ms = latency_ms
        self.judgements = judgements
        self.error = error

    @property
    def evaluated(self):
        """Did we get a verdict at all? A crash is a (failing) verdict; a case
        where every judge abstained is genuinely un-evaluated, not a failure."""
        if self.error is not None:
            return True
        return any(getattr(j, "applicable", True) for j in self.judgements)

    @property
    def passed(self):
        if self.error is not None:
            return False
        applicable = [j for j in self.judgements if getattr(j, "applicable", True)]
        return bool(applicable) and all(j.passed for j in applicable)

    def to_dict(self):
        return {"id": self.case.id, "input": self.case.input,
                "expected": self.case.expected, "output": self.output,
                "latency_ms": round(self.latency_ms, 2), "error": self.error,
                "passed": self.passed, "evaluated": self.evaluated,
                "judgements": [j.to_dict() for j in self.judgements]}


class Run:
    def __init__(self, version, results):
        self.version = version
        self.results = results

    def to_dict(self):
        return {"version": self.version,
                "results": [r.to_dict() for r in self.results]}


def run_eval(target, cases, judges, version="v1"):
    """target: callable(input) -> output (the system-under-test).
    judges: list[Judge]. A crashing target is recorded as a failed case, not a
    crash, so one bad case never sinks the whole run. Returns a Run."""
    results = []
    for case in cases:
        t0 = time.perf_counter()
        err = None
        try:
            output = target(case.input)
        except Exception as exc:
            output, err = "", "%s: %s" % (type(exc).__name__, exc)
        latency = (time.perf_counter() - t0) * 1000.0
        if err is None:
            judgements = [j.score(case, output) for j in judges]
        else:
            judgements = [Judgement(j.name, 0.0, False, err) for j in judges]
        results.append(CaseResult(case, output, latency, judgements, err))
    return Run(version, results)
