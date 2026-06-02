"""The `Embedder` seam.

Two methods, not one, on purpose. Retrieval-tuned cloud providers are
*asymmetric*: Voyage takes `input_type="document"` vs `"query"`, Cohere takes
`"search_document"` vs `"search_query"`. Using one symmetric `embed()` for both
sides silently degrades recall on those providers. Splitting the interface into
`embed_documents` / `embed_query` lets each implementation use the **same model**
on both sides with the correct per-side hint — which is the rule turbopuffer
cares about (the namespace's vector dimension and distance metric are fixed at
creation, so write-time and query-time embeddings must match exactly).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    #: Short, stable model identifier, used to derive a per-embedder namespace
    #: name (e.g. "bge-small-en-v1.5"). Different model+dim => different namespace.
    name: str
    #: Output dimensionality. Must equal the backend's declared vector dimension.
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents for indexing (write side)."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query (query side)."""
        ...
