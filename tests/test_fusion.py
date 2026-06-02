"""Unit tests for the shared reciprocal-rank fusion step.

RRF is the seam that makes "hybrid" mean the same thing on every backend:
turbopuffer has no server-side fusion, so each backend issues a vector sub-query
and a BM25 sub-query and fuses them *client-side* with this exact function. These
tests pin down the four properties the rest of the repo relies on — ordering,
dedup-by-id across lists, per-list weights, and the damping effect of `k` — using
small hand-built `Hit` lists (no embedder, no backend, no keys).
"""

from __future__ import annotations

import pytest

from retrieval_bridge.fusion import reciprocal_rank_fusion
from retrieval_bridge.types import Hit


def _hit(doc_id: str, text: str = "", metadata: dict | None = None) -> Hit:
    """Build a `Hit` with a placeholder score; RRF overwrites `score` anyway."""
    return Hit(id=doc_id, score=0.0, text=text, metadata=metadata or {})


def test_single_list_preserves_order_and_scores_descending() -> None:
    """A single ranked list keeps its order; fused scores strictly decrease and
    follow the RRF formula w/(k+rank) with the default k=60."""
    ranked = [_hit("a"), _hit("b"), _hit("c")]
    fused = reciprocal_rank_fusion([ranked])

    assert [h.id for h in fused] == ["a", "b", "c"]
    scores = [h.score for h in fused]
    assert scores == sorted(scores, reverse=True)
    assert fused[0].score == pytest.approx(1.0 / (60 + 1))
    assert fused[1].score == pytest.approx(1.0 / (60 + 2))
    assert fused[2].score == pytest.approx(1.0 / (60 + 3))


def test_dedup_by_id_across_lists() -> None:
    """An id appearing in two lists is emitted once, with its score summed across
    both lists' reciprocal ranks."""
    list_a = [_hit("a"), _hit("b"), _hit("c")]  # ranks 1,2,3
    list_b = [_hit("b"), _hit("a"), _hit("d")]  # ranks 1,2,3

    fused = reciprocal_rank_fusion([list_a, list_b])

    # Each id appears exactly once.
    ids = [h.id for h in fused]
    assert sorted(ids) == ["a", "b", "c", "d"]
    assert len(ids) == len(set(ids))

    by_id = {h.id: h for h in fused}
    # 'a': rank 1 in A + rank 2 in B ; 'b': rank 2 in A + rank 1 in B -> tie.
    expected_ab = 1.0 / (60 + 1) + 1.0 / (60 + 2)
    assert by_id["a"].score == pytest.approx(expected_ab)
    assert by_id["b"].score == pytest.approx(expected_ab)
    # 'c' (rank 3 in A only) and 'd' (rank 3 in B only) tie below a/b.
    expected_cd = 1.0 / (60 + 3)
    assert by_id["c"].score == pytest.approx(expected_cd)
    assert by_id["d"].score == pytest.approx(expected_cd)

    # Documents present in both lists rank above documents present in only one.
    assert set(ids[:2]) == {"a", "b"}
    assert set(ids[2:]) == {"c", "d"}


def test_representative_hit_from_earliest_list() -> None:
    """The emitted `Hit` for a shared id keeps the text/metadata from the first
    list that contained it (so payload is never lost on dedup)."""
    list_a = [_hit("a", text="from-A", metadata={"src": "A"})]
    list_b = [_hit("a", text="from-B", metadata={"src": "B"})]

    fused = reciprocal_rank_fusion([list_a, list_b])

    assert len(fused) == 1
    assert fused[0].text == "from-A"
    assert fused[0].metadata == {"src": "A"}
    # ...but the score is the fused sum across both occurrences.
    assert fused[0].score == pytest.approx(2.0 / (60 + 1))


def test_weights_upweight_one_modality() -> None:
    """Heavily weighting the second list promotes its top item above the first
    list's top item — e.g. upweighting BM25 for exact-token queries."""
    list_a = [_hit("a"), _hit("b"), _hit("c")]
    list_b = [_hit("b"), _hit("a"), _hit("d")]

    unweighted = reciprocal_rank_fusion([list_a, list_b])
    # With equal weights a and b tie; weighting list_b breaks the tie toward b.
    weighted = reciprocal_rank_fusion([list_a, list_b], weights=[1.0, 10.0])

    assert weighted[0].id == "b"
    # The unweighted result keeps a and b adjacent at the top (order-insensitive).
    assert set(h.id for h in unweighted[:2]) == {"a", "b"}


def test_k_damps_score_separation() -> None:
    """Smaller k sharpens the gap between consecutive ranks; larger k compresses
    it. Same ordering, different score spread."""
    ranked = [_hit("a"), _hit("b")]

    sharp = reciprocal_rank_fusion([ranked], k=1)
    flat = reciprocal_rank_fusion([ranked], k=1000)

    # Ordering is invariant to k.
    assert [h.id for h in sharp] == ["a", "b"]
    assert [h.id for h in flat] == ["a", "b"]

    sharp_gap = sharp[0].score - sharp[1].score
    flat_gap = flat[0].score - flat[1].score
    assert sharp_gap > flat_gap
    # Concrete values: k=1 -> 1/2 and 1/3 ; k=1000 -> 1/1001 and 1/1002.
    assert sharp[0].score == pytest.approx(1.0 / 2)
    assert flat[0].score == pytest.approx(1.0 / 1001)


def test_empty_lists() -> None:
    """Fusing nothing (or only-empty lists) yields an empty result, not an error."""
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_weights_length_must_match_lists() -> None:
    """A weights vector of the wrong length is a programming error and raises."""
    list_a = [_hit("a")]
    list_b = [_hit("b")]
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([list_a, list_b], weights=[1.0])
