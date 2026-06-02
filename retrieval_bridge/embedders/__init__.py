"""Embedder registry + factory.

Pick the embedder via the `RETRIEVAL_BRIDGE_EMBEDDER` env var (default: `local`).
The cloud embedders are optional extras; importing them is deferred so the base
install (local only) needs no cloud SDKs.
"""

from __future__ import annotations

import os

from .base import Embedder
from .local_bge import LocalBGEEmbedder

__all__ = ["Embedder", "LocalBGEEmbedder", "get_embedder"]


def get_embedder(name: str | None = None) -> Embedder:
    """Return an embedder instance.

    Args:
        name: one of "local", "openai", "voyage", "cohere". If None, reads
            `RETRIEVAL_BRIDGE_EMBEDDER` (default "local").
    """
    name = (name or os.getenv("RETRIEVAL_BRIDGE_EMBEDDER", "local")).lower()

    if name in ("local", "bge", "fastembed"):
        return LocalBGEEmbedder()
    if name == "openai":
        from .openai_embedder import OpenAIEmbedder

        return OpenAIEmbedder()
    if name == "voyage":
        from .voyage_embedder import VoyageEmbedder

        return VoyageEmbedder()
    if name == "cohere":
        from .cohere_embedder import CohereEmbedder

        return CohereEmbedder()
    raise ValueError(f"Unknown embedder: {name!r}")
