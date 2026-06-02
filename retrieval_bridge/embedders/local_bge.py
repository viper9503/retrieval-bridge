"""Zero-key local embedder: BAAI/bge-small-en-v1.5 via fastembed.

This is the default so the whole repo runs with **no API keys** — clone and go.
BGE-small (384-dim) is exactly the model turbopuffer uses in its own vector
search guide (https://turbopuffer.com/docs/vector), so the local default mirrors
the documented turbopuffer path; only the backend changes when you swap in the
real thing.

We use `fastembed` (ONNX runtime) rather than `sentence-transformers` to avoid
pulling in PyTorch — it keeps `pip install` light and CPU-only. The weights are
downloaded once on first use (~130 MB); after that it is fully offline.
"""

from __future__ import annotations

from functools import cached_property


class LocalBGEEmbedder:
    """fastembed BGE-small-en-v1.5. 384-dim, cosine. No API key required."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", dim: int = 384):
        self.name = "bge-small-en-v1.5"
        self.dim = dim
        self._model_name = model_name

    @cached_property
    def _model(self):
        # Imported lazily so that `import retrieval_bridge` is cheap and so the
        # model download only happens when embeddings are actually needed.
        from fastembed import TextEmbedding

        return TextEmbedding(model_name=self._model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        # fastembed's query_embed applies the model-appropriate query instruction.
        return list(next(self._model.query_embed(text)).tolist())
