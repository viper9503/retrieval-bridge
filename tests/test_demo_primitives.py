"""Tests for the offline stand-ins for PromptQL's `executor.*` primitives.

These deterministic stubs (`classify_root_cause` ~ `executor.classify`,
`summarize_fix` ~ `executor.summarize`) keep the demo zero-key and reproducible.
The tests assert the rule-driven behavior the demo and README cite: exact error
codes and HTTP statuses map to the expected canonical root-cause labels, and the
summarizer pulls the post-`Resolution:` narrative.
"""

from __future__ import annotations

import pytest

from retrieval_bridge.demo_primitives import classify_root_cause, summarize_fix


# --- classify_root_cause ---------------------------------------------------


@pytest.mark.parametrize(
    "text, expected_label",
    [
        # exact error-code tokens (the BM25 / exact-match story)
        ("Upsert failed with ERR_DIM_384 on the namespace", "dimension_mismatch"),
        ("vector dimension mismatch between query and index", "dimension_mismatch"),
        ("Pod was OOMKilled (137) during reindex", "out_of_memory"),
        ("ERR_TLS: certificate expired on the gateway", "expired_certificate"),
        ("ENOSPC: no space left on device", "disk_full"),
        ("ERR_POOL connection pool exhausted under load", "pool_exhausted"),
        # HTTP statuses / semantic phrasing
        ("503 Service Unavailable returned under load", "upstream_overload"),
        ("Service returned 429 Too Many Requests", "rate_limited"),
        ("401 Unauthorized after the SSO token expired", "expired_credential"),
        ("Replica showed stale, eventually-consistent reads", "replication_lag"),
        ("First query after idle hit a cold object-storage cache", "cold_cache"),
        # the not-an-incident escape hatch
        ("how do i rotate an api key", "question_not_incident"),
    ],
)
def test_classify_known_signals(text: str, expected_label: str) -> None:
    label, confidence = classify_root_cause(text)
    assert label == expected_label
    assert 0.0 < confidence <= 0.95


def test_classify_is_case_insensitive() -> None:
    """The same signal classifies identically regardless of case."""
    lower = classify_root_cause("err_dim_384 raised")
    upper = classify_root_cause("ERR_DIM_384 RAISED")
    assert lower[0] == upper[0] == "dimension_mismatch"


def test_classify_unknown_falls_through_with_low_confidence() -> None:
    """Text with no recognized signal returns the explicit 'unknown' label and a
    deliberately low confidence."""
    label, confidence = classify_root_cause("Customer asked about the new dashboard color palette.")
    assert label == "unknown"
    assert confidence == pytest.approx(0.3)


def test_classify_confidence_returns_float_in_range() -> None:
    label, confidence = classify_root_cause("503 503 503 saturated upstream overload")
    assert label == "upstream_overload"
    assert isinstance(confidence, float)
    assert confidence <= 0.95  # confidence is capped


def test_classify_specificity_order_first_match_wins() -> None:
    """The dimension-mismatch rule is more specific than the generic 'memory'
    rule; when both could appear, the earlier (more specific) rule wins."""
    label, _ = classify_root_cause("ERR_DIM_384 surfaced while the memory graph loaded")
    assert label == "dimension_mismatch"


# --- summarize_fix ---------------------------------------------------------


def test_summarize_extracts_text_after_resolution() -> None:
    text = (
        "Ticket: customers saw 503s after the 14:00 deploy.\n"
        "Investigation: the new pods overwhelmed the DB connection pool.\n"
        "Resolution: Raised the pool size to 50 and added a readiness probe."
    )
    summary = summarize_fix(text)
    assert summary == "Raised the pool size to 50 and added a readiness probe."


def test_summarize_collapses_whitespace_and_newlines() -> None:
    text = "Resolution:   Restarted   the\n   service\tcleanly."
    summary = summarize_fix(text)
    assert summary == "Restarted the service cleanly."


def test_summarize_without_marker_falls_back_to_body() -> None:
    """With no 'Resolution:' marker, it normalizes and returns the whole body."""
    text = "Just  some\nnarrative   without a marker."
    summary = summarize_fix(text)
    assert summary == "Just some narrative without a marker."


def test_summarize_truncates_long_text_with_ellipsis() -> None:
    """Over `max_chars`, the summary is truncated at a word boundary and gets a
    trailing single-character ellipsis."""
    text = "Resolution: " + ("word " * 100)
    summary = summarize_fix(text, max_chars=50)
    assert summary.endswith("…")
    # Truncation happens at a word boundary, so length stays within the budget.
    assert len(summary) <= 50
    # No partial trailing word before the ellipsis.
    assert summary[:-1].endswith("word")


def test_summarize_short_text_not_truncated() -> None:
    text = "Resolution: Bumped a flag."
    summary = summarize_fix(text, max_chars=200)
    assert summary == "Bumped a flag."
    assert "…" not in summary
