"""Backend registry + factory.

Pick the backend via `RETRIEVAL_BRIDGE_BACKEND` (default: `lancedb`). turbopuffer
and pgvector are optional extras; their SDKs are imported lazily so the base
install only needs LanceDB.
"""

from __future__ import annotations

import os

from .base import VectorBackend
from .lancedb_backend import LanceDBBackend

__all__ = ["VectorBackend", "LanceDBBackend", "get_backend"]


def get_backend(name: str | None = None, **kwargs) -> VectorBackend:
    """Return a backend instance.

    Args:
        name: one of "lancedb", "turbopuffer", "pgvector". If None, reads
            `RETRIEVAL_BRIDGE_BACKEND` (default "lancedb").
        **kwargs: forwarded to the backend constructor.
    """
    name = (name or os.getenv("RETRIEVAL_BRIDGE_BACKEND", "lancedb")).lower()

    if name in ("lancedb", "lance", "local"):
        return LanceDBBackend(**kwargs)
    if name in ("turbopuffer", "tpuf"):
        from .turbopuffer_backend import TurbopufferBackend

        return TurbopufferBackend(**kwargs)
    if name in ("pgvector", "postgres", "pg"):
        from .pgvector_backend import PgvectorBackend

        return PgvectorBackend(**kwargs)
    raise ValueError(f"Unknown backend: {name!r}")
