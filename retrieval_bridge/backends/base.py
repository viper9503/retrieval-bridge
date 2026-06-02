"""The `VectorBackend` seam — the heart of the swappable-backend pitch.

Every backend takes the *same* inputs and returns the *same* `list[Hit]`, so the
PromptQL-facing `search_documents` command never changes when you swap the store.
`search` receives both the query embedding (for the vector sub-query) and the raw
query text (for the BM25 / full-text sub-query); each backend runs both and fuses
them with the shared `reciprocal_rank_fusion`.

This is deliberately the same surface turbopuffer exposes (vector + BM25 +
client-side RRF), so `TurbopufferBackend` is a faithful default and `LanceDBBackend`
/ `PgvectorBackend` are honest stand-ins, not toys.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..types import Hit


@runtime_checkable
class VectorBackend(Protocol):
    #: Human-readable backend id, for logging/diagnostics ("turbopuffer", ...).
    name: str

    def upsert(self, docs: list[dict[str, Any]]) -> None:
        """Index documents.

        Each doc is a dict with keys: `id` (str), `vector` (list[float]),
        `text` (str), and any number of scalar metadata attributes
        (project_id, plan_tier, status, created_at, ...).
        """
        ...

    def search(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[Hit]:
        """Hybrid search: vector ANN + full-text BM25, fused with RRF.

        `filters` is a backend-agnostic dict of equality constraints on metadata
        attributes, e.g. {"status": "resolved", "plan_tier": "enterprise"}.
        """
        ...
