"""Offline stand-ins for PromptQL's LLM primitives.

IMPORTANT: these are **not** the real primitives. In the deployed architecture,
the generated PromptQL plan calls `executor.classify(...)` and
`executor.summarize(...)` — LLM-backed functions that run in isolated, focused
context inside the deterministic PromptQL runtime. To keep the standalone demo
zero-key and reproducible, we emulate them here with simple deterministic logic
and label every call site accordingly.

The point of the demo is the *shape* of the plan (retrieve -> classify ->
summarize -> join -> rank), not that these stand-ins are clever. The README maps
each one to its real `executor.*` counterpart.
"""

from __future__ import annotations

import re

# Keyword -> canonical root-cause label. First match wins (ordered by specificity).
_RULES: list[tuple[str, str]] = [
    (r"err_dim|dimension|vector dimension", "dimension_mismatch"),
    (r"err_oom|oomkilled|137|out[- ]of[- ]memory|memory", "out_of_memory"),
    (r"err_tls|526|certificate|cert\b|tls", "expired_certificate"),
    (r"err_disk|enospc|no space|disk", "disk_full"),
    (r"err_pool|pool exhausted|connection pool", "pool_exhausted"),
    (r"429|throttl|rate limit|too many requests", "rate_limited"),
    (r"401|419|err_auth|sso|token|session expired", "expired_credential"),
    (r"\blag\b|stale|eventually[- ]consistent|read-your-writes", "replication_lag"),
    (r"cold|idle|warm|object storage|first query|first request", "cold_cache"),
    (r"503|unavailable|saturat|overload|shed load", "upstream_overload"),
    (r"how do i|question|not an incident", "question_not_incident"),
]


def classify_root_cause(text: str) -> tuple[str, float]:
    """Stand-in for `executor.classify(...)`. Returns (label, confidence)."""
    t = text.lower()
    for pattern, label in _RULES:
        if re.search(pattern, t):
            # crude confidence: more distinct signal tokens -> higher confidence
            hits = len(re.findall(pattern, t))
            return label, min(0.6 + 0.1 * hits, 0.95)
    return "unknown", 0.3


def summarize_fix(text: str, max_chars: int = 200) -> str:
    """Stand-in for `executor.summarize(...)`. Pulls the resolution narrative."""
    m = re.search(r"Resolution:\s*(.+)", text, flags=re.DOTALL)
    snippet = (m.group(1) if m else text).strip().replace("\n", " ")
    snippet = re.sub(r"\s+", " ", snippet)
    if len(snippet) <= max_chars:
        return snippet
    cut = snippet[:max_chars].rsplit(" ", 1)[0]
    return cut + "…"
