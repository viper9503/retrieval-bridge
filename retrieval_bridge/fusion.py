"""Reciprocal-rank fusion (RRF).

turbopuffer has **no server-side rank fusion** (verified against
https://turbopuffer.com/docs/hybrid): hybrid search is done by issuing a
multi-query (one vector sub-query + one BM25 sub-query) against a single
consistent snapshot, then fusing the two ranked lists *client-side*. This module
is that fusion step, and every backend in this repo reuses it so that "hybrid"
means exactly the same thing whether the store is turbopuffer, LanceDB, or
pgvector. That uniformity is the whole point of the swappable-backend design.

RRF score for a document d:  sum over result lists L of  w_L / (k + rank_L(d))
where rank is 1-based and k (default 60) damps the influence of low ranks.
"""

from __future__ import annotations

from .types import Hit


def reciprocal_rank_fusion(
    ranked_lists: list[list[Hit]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[Hit]:
    """Fuse several ranked lists of `Hit` into one, by reciprocal rank.

    Args:
        ranked_lists: each inner list is one modality's results, best-first.
        k: RRF damping constant (60 is the canonical default).
        weights: optional per-list weights (e.g. upweight BM25 for exact tokens).
            Defaults to 1.0 for every list.

    Returns:
        A single list of `Hit`, best-first, with `score` set to the fused score.
        The representative `Hit` object for each id is taken from the earliest
        list that contained it (so its text/metadata are preserved).
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must have one entry per ranked list")

    fused_score: dict[str, float] = {}
    representative: dict[str, Hit] = {}

    for weight, results in zip(weights, ranked_lists):
        for rank, hit in enumerate(results, start=1):
            fused_score[hit.id] = fused_score.get(hit.id, 0.0) + weight / (k + rank)
            representative.setdefault(hit.id, hit)

    ordered_ids = sorted(fused_score, key=lambda i: fused_score[i], reverse=True)
    out: list[Hit] = []
    for doc_id in ordered_ids:
        hit = representative[doc_id]
        out.append(hit.model_copy(update={"score": fused_score[doc_id]}))
    return out
