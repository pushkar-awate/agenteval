"""agentcore: a small, from-scratch agent core.

Task-agnostic building blocks: a perceive-reason-guardrail-act-verify-remember
loop, a pluggable reasoning brain (deterministic by default, real LLM when you
want it), a tool registry, an append-only decision ledger, and guardrails. Swap
the tool set and the same core drives a different agent - that reuse is the
point. It already powers jobfit-agent and the selfheal-mlops controller; in
agenteval the brain is the reasoning backend behind the LLM-as-judge.
"""
