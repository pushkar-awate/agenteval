"""A tiny JSONL response cache.

LLM judges cost money and rate-limit. Caching their answers keyed by
(judge, prompt) means a re-run of the same eval doesn't re-bill the API or trip
a 429 - the same input always maps to the same stored judgement. Append-only
JSONL so it survives across runs and is trivial to inspect or delete.
"""
import hashlib
import json
import os


class JSONLCache:
    def __init__(self, path=None):
        self.path = path
        self._mem = {}
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        self._mem[d["key"]] = d["value"]
                    except Exception:
                        pass  # a corrupt line never breaks a run

    @staticmethod
    def key(*parts):
        raw = "\x00".join(str(p) for p in parts).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def get(self, key):
        return self._mem.get(key)

    def put(self, key, value):
        self._mem[key] = value
        if self.path:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "value": value}) + "\n")
        return value

    def __len__(self):
        return len(self._mem)

    def __bool__(self):
        # a cache with __len__ would otherwise be "falsy" when empty, which
        # silently disables caching on the very first (empty) run. It is a real
        # object; treat it as truthy.
        return True
