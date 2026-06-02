"""OpenAI embedder — a pluggable cloud `Embedder` for the retrieval seam.

OpenAI's embedding models are **symmetric**: there is no document-vs-query input
hint, so both sides go through the identical `embeddings.create` call. We still
honour the two-method `Embedder` contract (it is the lowest common denominator
across providers), they just dispatch to the same place here.

This swaps in behind the same `search_documents` command that turbopuffer-style
hybrid retrieval exposes: the embedder only produces the dense vector, and the
backend (turbopuffer / LanceDB / pgvector) does the ANN + BM25 + client-side RRF.
Because a turbopuffer namespace pins its vector dimension at creation, the model
and `dim` chosen here MUST match the backend's declared vector dimension, and a
different model+dim implies a different namespace. `text-embedding-3-small`
(default, 1536-dim) and `-3-large` (3072-dim) both accept an optional
`dimensions=` to shrink the output (Matryoshka), which we surface via the ctor.

SDK is imported lazily so the base (local-only) install needs no `openai`; the
API key is read from `OPENAI_API_KEY` and a missing key fails loudly.
"""

from __future__ import annotations

import os
from functools import cached_property


# Native output dimensions per model. `dimensions=` (Matryoshka shrink) is only
# accepted by the -3 models, and must be sent whenever the requested dim differs
# from the model's native size — otherwise e.g. -3-large silently returns 3072
# while self.dim claims 1536, breaking the namespace dimension contract.
_NATIVE_DIM = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}


class OpenAIEmbedder:
    """OpenAI text embeddings. Symmetric (same call for documents and queries)."""

    def __init__(
        self,
        model: str | None = None,
        dim: int | None = None,
    ):
        # Model + dim are configurable via ctor or env. Default model is
        # text-embedding-3-small at its native 1536 dims.
        self._model = model or os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        env_dim = os.getenv("OPENAI_EMBED_DIM")
        self.dim = dim if dim is not None else (int(env_dim) if env_dim else 1536)
        self.name = f"openai-{self._model}"

    @cached_property
    def _client(self):
        # Lazy import + lazy construction: keeps `import retrieval_bridge` cheap
        # and means the SDK is only required when this embedder is actually used.
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OpenAIEmbedder requires the OPENAI_API_KEY environment variable."
            )
        from openai import OpenAI

        return OpenAI()  # reads OPENAI_API_KEY from the environment

    def _embed(self, texts: list[str]) -> list[list[float]]:
        # Pass `dimensions` whenever the requested dim differs from THIS model's
        # native size (gate on the model, not a hardcoded pair) so the returned
        # vectors always match self.dim. If a model's native dim is unknown, send
        # `dimensions` to be safe (the -3 models accept it).
        kwargs: dict = {"model": self._model, "input": texts}
        native = _NATIVE_DIM.get(self._model)
        if native is None or self.dim != native:
            kwargs["dimensions"] = self.dim
        resp = self._client.embeddings.create(**kwargs)
        return [item.embedding for item in resp.data]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(list(texts))

    def embed_query(self, text: str) -> list[float]:
        # Symmetric: the query side uses the exact same call as the write side.
        return self._embed([text])[0]
