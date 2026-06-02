"""End-to-end-ish test of the LanceDB backend — the zero-key clone-and-run path.

This exercises the real seam (`upsert` -> table + BM25 FTS index, then hybrid
`search` = vector ANN + BM25 fused with the shared RRF) without any cloud keys,
without fastembed, and without network: we feed tiny deterministic seeded 8-dim
vectors through a fake embedder (plain Python lists) and write them straight into
a temporary LanceDB at `tmp_path`. The assertions are intentionally tolerant of
LanceDB minor-version API differences; if LanceDB isn't installed the module is
skipped rather than failing the suite.
"""

from __future__ import annotations

import random

import pytest

# LanceDB is the default backend but still an optional dependency for the base
# install; skip the whole module cleanly if it (or pyarrow) is unavailable.
pytest.importorskip("lancedb")

from retrieval_bridge.backends.lancedb_backend import LanceDBBackend  # noqa: E402
from retrieval_bridge.types import Hit  # noqa: E402

DIM = 8


class FakeEmbedder:
    """Deterministic, network-free embedder: seeded 8-dim float lists.

    Satisfies the `Embedder` shape (name/dim/embed_documents/embed_query) but
    never touches a model — the point is to test the backend wiring, not recall.
    """

    name = "fake-8d"
    dim = DIM

    def __init__(self, seed: int = 1234) -> None:
        self._rng = random.Random(seed)

    def _vec(self) -> list[float]:
        return [self._rng.uniform(-1.0, 1.0) for _ in range(self.dim)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec() for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec()


def _build_docs(embedder: FakeEmbedder) -> list[dict]:
    """Six tiny docs in the backend's upsert shape: id, text, vector, + scalar
    metadata (status / plan_tier) that the filter test will constrain on."""
    rows = [
        ("t1", "503 service unavailable after deploy", "resolved", "scale"),
        ("t2", "timeouts under load on the api gateway", "resolved", "enterprise"),
        ("t3", "ERR_DIM_384 vector dimension mismatch", "open", "launch"),
        ("t4", "connection pool exhausted under heavy load", "resolved", "scale"),
        ("t5", "how do i rotate an api key", "open", "launch"),
        ("t6", "certificate expired on the edge gateway", "resolved", "enterprise"),
    ]
    texts = [r[1] for r in rows]
    vectors = embedder.embed_documents(texts)
    docs: list[dict] = []
    for (doc_id, text, status, plan_tier), vector in zip(rows, vectors):
        docs.append(
            {
                "id": doc_id,
                "text": text,
                "vector": vector,
                "status": status,
                "plan_tier": plan_tier,
            }
        )
    return docs


@pytest.fixture()
def seeded_backend(tmp_path) -> tuple[LanceDBBackend, FakeEmbedder]:
    """A LanceDBBackend rooted at an isolated tmp dir, pre-seeded with 6 docs."""
    embedder = FakeEmbedder()
    backend = LanceDBBackend(uri=str(tmp_path / "lancedb"), table="t")
    backend.upsert(_build_docs(embedder))
    return backend, embedder


def test_upsert_is_noop_on_empty(tmp_path) -> None:
    """Upserting nothing must not create a table or raise."""
    backend = LanceDBBackend(uri=str(tmp_path / "empty"), table="t")
    backend.upsert([])  # should simply return


def test_search_returns_hits(seeded_backend) -> None:
    backend, embedder = seeded_backend
    results = backend.search(
        query_embedding=embedder.embed_query("503 after deploy under load"),
        query_text="503 after deploy under load",
        top_k=4,
    )

    assert isinstance(results, list)
    assert results, "hybrid search should return at least one Hit"
    assert all(isinstance(h, Hit) for h in results)
    assert len(results) <= 4  # top_k is respected

    # Hits are well-formed: real ids from the corpus, no duplicates, scores set.
    known_ids = {"t1", "t2", "t3", "t4", "t5", "t6"}
    ids = [h.id for h in results]
    assert set(ids) <= known_ids
    assert len(ids) == len(set(ids))  # RRF dedups across the two sub-queries
    assert all(isinstance(h.score, float) for h in results)
    assert all(isinstance(h.text, str) and h.text for h in results)


def test_bm25_finds_exact_token(seeded_backend) -> None:
    """The full-text half of the hybrid query should surface the doc carrying the
    exact rare token, demonstrating the keyword side really runs."""
    backend, embedder = seeded_backend
    results = backend.search(
        query_embedding=embedder.embed_query("ERR_DIM_384"),
        query_text="ERR_DIM_384",
        top_k=6,
    )
    assert "t3" in {h.id for h in results}


def test_filter_restricts_results(seeded_backend) -> None:
    """An equality filter must constrain results to matching metadata only."""
    backend, embedder = seeded_backend

    resolved = backend.search(
        query_embedding=embedder.embed_query("gateway under load"),
        query_text="gateway under load",
        top_k=6,
        filters={"status": "resolved"},
    )
    assert resolved, "filtered search should still return matches"
    assert all(h.metadata.get("status") == "resolved" for h in resolved)
    # None of the open-status docs leak through.
    assert {"t3", "t5"}.isdisjoint({h.id for h in resolved})


def test_in_filter_membership(seeded_backend) -> None:
    """A list-valued filter is treated as IN (...) membership."""
    backend, embedder = seeded_backend
    results = backend.search(
        query_embedding=embedder.embed_query("load and gateways"),
        query_text="load and gateways",
        top_k=6,
        filters={"plan_tier": ["scale", "enterprise"]},
    )
    assert results
    assert all(h.metadata.get("plan_tier") in {"scale", "enterprise"} for h in results)


def test_search_carries_metadata(seeded_backend) -> None:
    """Scalar metadata round-trips through upsert/search and excludes the raw
    vector / internal score columns."""
    backend, embedder = seeded_backend
    results = backend.search(
        query_embedding=embedder.embed_query("anything"),
        query_text="anything",
        top_k=6,
    )
    assert results
    sample = results[0]
    assert "status" in sample.metadata
    assert "plan_tier" in sample.metadata
    # Reserved / structural columns must not bleed into metadata.
    for reserved in ("vector", "_distance", "_score", "_relevance_score", "_rowid", "id", "text"):
        assert reserved not in sample.metadata
