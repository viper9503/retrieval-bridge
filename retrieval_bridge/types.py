"""Shared data contracts for the retrieval bridge.

`Hit` is the single object that crosses every seam in this project:

    backend.search(...) -> list[Hit] -> RetrievalBridge.search_documents(...)
                                     -> the DDN `search_documents` command
                                     -> PromptQL artifact rows

Keeping it a small, explicit pydantic model (not a bare dict) matters for the
Hasura DDN integration: the lambda connector's return type must be a *named
object type* with a concrete `id` field for the command-result -> Postgres-model
relationship to attach (see ddn/metadata/). The same shape therefore serves both
the local demo and the deployed connector.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Hit(BaseModel):
    """A single retrieval result, backend-agnostic.

    `score` is the *fused* relevance score (higher is better) produced by
    reciprocal-rank fusion over the vector and full-text sub-queries. It is not
    comparable across backends in absolute terms, only as a ranking within one
    result set.
    """

    id: str = Field(description="Stable document id; the join key to structured data.")
    score: float = Field(description="Fused relevance score; higher is better.")
    text: str = Field(description="The retrieved document text (e.g. the ticket body).")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Filterable attributes: project_id, plan_tier, status, created_at, ...",
    )

    def __hash__(self) -> int:  # allow use in sets / dict keys by id
        return hash(self.id)


class Document(BaseModel):
    """An input document to index. `vector` is filled in by the embedder before
    it reaches a backend's `upsert`."""

    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    vector: list[float] | None = None
