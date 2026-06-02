"""Voyage embedder — a pluggable cloud `Embedder` for the retrieval seam.

Voyage's retrieval models are **asymmetric**: the *same* model is called with
`input_type="document"` at index time and `input_type="query"` at search time.
This is exactly why the `Embedder` contract splits into `embed_documents` /
`embed_query` — collapsing them into one symmetric call would silently drop
recall on Voyage. Both sides must use the same model and dimension, since a
turbopuffer namespace pins its vector dimension at creation; pick a model+dim
here that matches the backend's declared vector dimension (a different model+dim
implies a different namespace).

This plugs in behind the same `search_documents` command that turbopuffer-style
hybrid retrieval exposes: the embedder only yields the dense vector, and the
backend (turbopuffer / LanceDB / pgvector) runs ANN + BM25 with client-side RRF
(turbopuffer has no server-side fusion).

SDK is imported lazily so the base install needs no `voyageai`; the API key is
read from `VOYAGE_API_KEY` and a missing key fails loudly.
"""

from __future__ import annotations

import os
from functools import cached_property


class VoyageEmbedder:
    """Voyage AI embeddings. Asymmetric (document vs query input_type)."""

    def __init__(
        self,
        model: str | None = None,
        dim: int | None = None,
    ):
        # Default to voyage-3.5 (1024-dim). voyage-3 / voyage-3-lite are
        # deprecated and intentionally not defaulted to. `dim` is exposed for
        # models/configs that support output_dimension.
        self._model = model or os.getenv("VOYAGE_EMBED_MODEL", "voyage-3.5")
        env_dim = os.getenv("VOYAGE_EMBED_DIM")
        self.dim = dim if dim is not None else (int(env_dim) if env_dim else 1024)
        # name uses the model suffix after "voyage-" (e.g. "voyage-3.5").
        suffix = self._model[len("voyage-"):] if self._model.startswith("voyage-") else self._model
        self.name = f"voyage-{suffix}"

    @cached_property
    def _client(self):
        # Lazy import + lazy construction so the base install stays SDK-free.
        if not os.getenv("VOYAGE_API_KEY"):
            raise RuntimeError(
                "VoyageEmbedder requires the VOYAGE_API_KEY environment variable."
            )
        import voyageai

        return voyageai.Client()  # reads VOYAGE_API_KEY from the environment

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        kwargs: dict = {"texts": texts, "model": self._model, "input_type": input_type}
        # Only request a non-native dimension when the caller asked for one.
        if self.dim != 1024:
            kwargs["output_dimension"] = self.dim
        return self._client.embed(**kwargs).embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Write side: documents get the "document" hint.
        return self._embed(list(texts), input_type="document")

    def embed_query(self, text: str) -> list[float]:
        # Query side: same model, but the "query" hint — this is the asymmetry.
        return self._embed([text], input_type="query")[0]
