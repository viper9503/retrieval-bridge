"""Cohere embedder — a pluggable cloud `Embedder` for the retrieval seam.

Cohere's retrieval models are **asymmetric**: the same model is called with
`input_type="search_document"` at index time and `input_type="search_query"` at
search time. The `Embedder` contract splits into `embed_documents` /
`embed_query` precisely so this per-side hint survives — a single symmetric call
would silently degrade recall. Both sides use the same model and dimension,
because a turbopuffer namespace pins its vector dimension at creation; the `dim`
chosen here MUST match the backend's declared vector dimension (different
model+dim => different namespace).

This plugs in behind the same `search_documents` command that turbopuffer-style
hybrid retrieval exposes: the embedder only produces the dense vector, and the
backend (turbopuffer / LanceDB / pgvector) handles ANN + BM25 with client-side
RRF (turbopuffer has no server-side fusion).

SDK is imported lazily so the base install needs no `cohere`; the API key is read
from `CO_API_KEY` and a missing key fails loudly.
"""

from __future__ import annotations

import os
from functools import cached_property


class CohereEmbedder:
    """Cohere embed-v4.0 embeddings. Asymmetric (search_document vs search_query)."""

    def __init__(
        self,
        model: str = "embed-v4.0",
        dim: int | None = None,
    ):
        # embed-v4.0 is Matryoshka: supports 256/512/1024/1536 (default 1024).
        self._model = model
        env_dim = os.getenv("COHERE_EMBED_DIM")
        self.dim = dim if dim is not None else (int(env_dim) if env_dim else 1024)
        self.name = f"cohere-{self._model}"

    @cached_property
    def _client(self):
        # Lazy import + lazy construction keeps the base install SDK-free.
        if not os.getenv("CO_API_KEY"):
            raise RuntimeError(
                "CohereEmbedder requires the CO_API_KEY environment variable."
            )
        import cohere

        return cohere.ClientV2()  # reads CO_API_KEY from the environment

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        resp = self._client.embed(
            model=self._model,
            input_type=input_type,
            texts=texts,
            embedding_types=["float"],
            output_dimension=self.dim,
        )
        return resp.embeddings.float

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Write side: documents get the "search_document" hint.
        return self._embed(list(texts), input_type="search_document")

    def embed_query(self, text: str) -> list[float]:
        # Query side: same model, but the "search_query" hint — the asymmetry.
        return self._embed([text], input_type="search_query")[0]
