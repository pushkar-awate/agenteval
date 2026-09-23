"""A small system-under-test: classify a support message into an intent.

This is the "agent" the eval engine scores. It is deterministic on purpose so
the scaffold runs with no API key and the tests are reproducible. Later
milestones swap in an agentcore-based agent and an LLM-backed target - the eval
engine does not change, only the callable it points at.
"""

_RULES = [
    ("billing", ["refund", "invoice", "charged", "charge", "payment", "billing",
                 "subscription", "price", "credit card"]),
    ("technical", ["error", "bug", "crash", "not working", "broken", "fails",
                   "password reset", "500"]),
    ("account", ["email address", "username", "delete my account", "my account",
                 "profile", "sign up", "register"]),
    ("shipping", ["delivery", "shipping", "track", "package", "order status",
                  "arrive", "arrived"]),
]


def classify(text):
    """Return the intent label for a support message."""
    low = text.lower()
    for label, keywords in _RULES:
        if any(k in low for k in keywords):
            return label
    return "other"
