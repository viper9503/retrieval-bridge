"""High-level orchestration: embedder + backend = the retrieval bridge.

`RetrievalBridge.search_documents` is the exact function the Hasura DDN lambda
connector wraps and exposes to PromptQL as a command (see
ddn/connector/search/functions.py). Keeping it here, backend- and
embedder-agnostic, is what lets the same logic run in the local demo and in the
deployed connector.
"""

from __future__ import annotations

from typing import Any

from .backends import VectorBackend, get_backend
from .embedders import Embedder, get_embedder
from .types import Hit


class RetrievalBridge:
    def __init__(self, backend: VectorBackend | None = None, embedder: Embedder | None = None):
        self.embedder = embedder or get_embedder()
        self.backend = backend or get_backend()

    # ---- write path -------------------------------------------------------
    def index(self, docs: list[dict[str, Any]], batch_size: int = 256) -> int:
        """Embed `docs` (each must have `id`, `text`, and optional metadata
        attributes) and upsert them into the backend. Returns the count."""
        prepared: list[dict[str, Any]] = []
        for start in range(0, len(docs), batch_size):
            batch = docs[start : start + batch_size]
            vectors = self.embedder.embed_documents([d["text"] for d in batch])
            for doc, vec in zip(batch, vectors):
                row = {k: v for k, v in doc.items() if k != "vector"}
                row["vector"] = vec
                prepared.append(row)
        self.backend.upsert(prepared)
        return len(prepared)

    # ---- read path (the PromptQL-facing command) --------------------------
    def search_documents(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[Hit]:
        """Hybrid (semantic + keyword) search over the indexed corpus.

        Embeds the query once, then asks the backend for a fused vector+BM25
        result. This is intentionally the whole retrieval step — narrowing the
        corpus to a handful of strong candidates — after which PromptQL does the
        planning, joining, and reasoning.
        """
        query_embedding = self.embedder.embed_query(query)
        return self.backend.search(query_embedding, query, top_k=top_k, filters=filters)
