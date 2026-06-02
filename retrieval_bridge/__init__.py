"""retrieval-bridge: turbopuffer-style hybrid retrieval as a swappable backend
for PromptQL, behind a single stable command interface.

Public surface:
    from retrieval_bridge import RetrievalBridge, get_backend, get_embedder, Hit
"""

from __future__ import annotations

from .backends import VectorBackend, get_backend
from .embedders import Embedder, get_embedder
from .fusion import reciprocal_rank_fusion
from .search import RetrievalBridge
from .structured import StructuredStore
from .types import Document, Hit

__all__ = [
    "RetrievalBridge",
    "Hit",
    "Document",
    "VectorBackend",
    "Embedder",
    "get_backend",
    "get_embedder",
    "reciprocal_rank_fusion",
    "StructuredStore",
]

__version__ = "0.1.0"
