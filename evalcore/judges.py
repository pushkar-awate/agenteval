"""Judges score one (case, output) pair.

Two families:

  * a deterministic "mock" family - ExactMatch, Contains, Regex, FormatAdherence,
    GuardrailJudge - rule-based, need no API key, and keep tests reproducible
    (mirroring agentcore's MockBrain).
  * an LLM-as-judge - LLMJudge - that reuses *any* agentcore Brain via
    Brain.complete(prompt) to score fuzzy qualities like faithfulness and
    relevance. It caches responses and caps input size, so re-runs don't re-bill
    the API. With the deterministic MockBrain it simply abstains, so the default
    key-free path never depends on it.

Every judge returns a Judgement(name, score in [0,1], passed, detail, meta).
"""
import json
import re

from evalcore.cache import JSONLCache

try:                                    # reuse agentcore's robust JSON extractor
    from agentcore.brain import extract_json
except Exception:                       # keep evalcore importable on its own
    def extract_json(text):
        start, end = text.find("{"), text.rfind("}")
        return text[start:end + 1] if start >= 0 and end > start else text


def _est_tokens(text):
    """A rough token estimate (~4 chars/token) - enough for a cost ballpark."""
    return max(0, len(str(text)) // 4)


class Judgement:
    def __init__(self, name, score, passed, detail="", meta=None, applicable=True):
        self.name = name
        self.score = float(score)
        self.passed = bool(passed)
        # applicable=False means this judge could not render a verdict (e.g. an
        # LLM judge whose call failed). It is neutral: excluded from task success
        # and from pass-rate denominators, never counted as a failure.
        self.applicable = bool(applicable)
        self.detail = detail
        self.meta = meta or {}

    def to_dict(self):
        return {"name": self.name, "score": self.score, "passed": self.passed,
                "applicable": self.applicable, "detail": self.detail,
                "meta": self.meta}


class Judge:
    name = "judge"

    def score(self, case, output):
        raise NotImplementedError


def _norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())


# --- deterministic judges (no key) ----------------------------------------

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


class FormatAdherence(Judge):
    """Does the whole output match a required shape (a label set, a JSON line, an
    id format)? Deterministic - a cheap proxy for 'the agent returned something
    the downstream system can parse'."""
    name = "format_adherence"

    def __init__(self, pattern, name=None):
        self.pattern = re.compile(pattern)
        if name:
            self.name = name

    def score(self, case, output):
        ok = bool(self.pattern.fullmatch(str(output).strip()))
        return Judgement(self.name, 1.0 if ok else 0.0, ok,
                         "" if ok else "output %r fails the required format" % output)


def _default_guardrail(output):
    if not output.strip():
        return False, "empty output"
    low = output.lower()
    for phrase in ("as an ai language model", "i cannot help with that",
                   "i'm sorry, but i can't"):
        if phrase in low:
            return False, "contains guardrail-tripping phrase"
    return True, ""


class GuardrailJudge(Judge):
    """Guardrail pass rate: does the output satisfy a safety/quality guardrail?
    Takes a validator(output) -> (ok, reason); the default rejects empty or
    boiler-plate refusal answers. This is the agent-side echo of selfheal's
    deploy guardrail."""
    name = "guardrail"

    def __init__(self, validator=None, name=None):
        self.validator = validator or _default_guardrail
        if name:
            self.name = name

    def score(self, case, output):
        ok, reason = self.validator(str(output))
        return Judgement(self.name, 1.0 if ok else 0.0, ok, "" if ok else reason)


# --- LLM-as-judge (reuses agentcore Brain) --------------------------------

class LLMJudge(Judge):
    """Score an output with an LLM through any agentcore Brain.

    brain.complete(prompt) does the work, so a GroqBrain scores for real, a stub
    scores in tests, and the deterministic MockBrain (which returns "") makes the
    judge *abstain* - it never silently passes. Responses are cached by
    (judge, prompt) and inputs are truncated, so re-runs are free and bounded.
    """
    name = "llm_judge"

    def __init__(self, brain, rubric, name=None, threshold=0.6, cache=None,
                 max_chars=4000):
        self.brain = brain
        self.rubric = rubric
        self.threshold = threshold
        self.cache = cache
        self.max_chars = max_chars
        if name:
            self.name = name

    def _prompt(self, case, output):
        return (
            "You are an impartial evaluator. %s\n"
            'Return ONLY JSON: {"score": <0-100 integer>, "reason": <one short sentence>}.\n\n'
            "INPUT:\n%s\n\nEXPECTED (reference or rubric):\n%s\n\nAGENT OUTPUT:\n%s"
            % (self.rubric, str(case.input)[:self.max_chars],
               str(case.expected)[:self.max_chars], str(output)[:self.max_chars]))

    def score(self, case, output):
        prompt = self._prompt(case, output)
        key = JSONLCache.key(self.name, prompt) if self.cache else None
        raw = self.cache.get(key) if (self.cache and key) else None
        cached = raw is not None
        if raw is None:
            raw = self.brain.complete(prompt) or ""
            if self.cache and key and raw:
                self.cache.put(key, raw)
        if not raw:                       # MockBrain / failed call -> abstain (neutral)
            return Judgement(self.name, 0.0, False,
                             "no LLM response (this judge needs a real brain)",
                             meta={"abstained": True}, applicable=False)
        try:
            d = json.loads(extract_json(raw))
            raw_score = float(d.get("score", 0))
            # accept either a 0-100 scale (the asked-for format) or a 0-1 scale
            # some models return anyway; <=1 is read as a fraction, else /100.
            score = raw_score if raw_score <= 1.0 else raw_score / 100.0
            score = max(0.0, min(1.0, score))
            passed = score >= self.threshold
            # tokens/cost count only a REAL call, not a cache hit (the cache's
            # whole point is not to re-bill).
            meta = {"cached": cached}
            if not cached:
                meta["est_tokens"] = _est_tokens(prompt) + _est_tokens(raw)
            return Judgement(self.name, score, passed,
                             str(d.get("reason", ""))[:200], meta=meta)
        except Exception:
            # an unparseable response is not a failed agent - it is a failed
            # judgement. Neutral, so a broken judge never fakes a regression.
            return Judgement(self.name, 0.0, False, "unparseable LLM response",
                             meta={"cached": cached}, applicable=False)


def faithfulness_judge(brain, cache=None, threshold=0.6):
    return LLMJudge(brain,
        "Judge FAITHFULNESS: does the agent output stay true to the input and the "
        "expected reference, without inventing facts or contradicting them?",
        name="faithfulness", threshold=threshold, cache=cache)


def relevance_judge(brain, cache=None, threshold=0.6):
    return LLMJudge(brain,
        "Judge RELEVANCE: does the agent output actually address what the input "
        "asks - on-topic, complete, and not padded with irrelevant content?",
        name="relevance", threshold=threshold, cache=cache)
