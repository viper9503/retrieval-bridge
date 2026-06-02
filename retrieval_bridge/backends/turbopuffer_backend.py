"""turbopuffer backend — the headline swappable PromptQL retrieval store.

turbopuffer is an object-storage-native vector + full-text database. It is the
default backend this artifact is pitched around, and it is wired behind the very
same `VectorBackend` seam as LanceDB and pgvector: the PromptQL-facing
`search_documents` command never changes when you point it here.

The hybrid recipe is turbopuffer's own, done by the book:

  * `upsert` declares a schema with a `[dim]f32` vector column (`ann: true`) and a
    `text` column flagged `full_text_search` (BM25 requires that flag); every other
    attribute is filterable by default, so no schema entry is needed for filters.
  * `search` issues a single snapshot-isolated `multi_query` with two sub-queries —
    a vector ANN rank and a BM25 text rank — because **turbopuffer has no
    server-side rank fusion**. The two ranked lists come back and are fused
    *client-side* with the shared `reciprocal_rank_fusion`, exactly as the other
    backends do. That is the whole point: "hybrid" means the same thing everywhere.

Operationally turbopuffer is cold/warm tiered (cold reads from object storage land
in the hundreds of ms; warm namespaces serve in ~14ms), which is why the demo
corpus is small and re-queried. The SDK is imported lazily so the base install
never needs it, and the API key / region / namespace are read from the
environment.
"""

from __future__ import annotations

import os
from typing import Any

from ..fusion import reciprocal_rank_fusion
from ..types import Hit

#: Attributes that are turbopuffer plumbing, not user metadata, and must be
#: stripped before a row becomes a `Hit`.
_RESERVED = {"id", "text", "vector", "$dist", "$score", "_dist", "_score"}

#: Map the backend-agnostic operator-dict keys onto turbopuffer's tuple-DSL ops.
_OP_MAP = {
    "eq": "Eq",
    "neq": "NotEq",
    "ne": "NotEq",
    "gt": "Gt",
    "gte": "Gte",
    "lt": "Lt",
    "lte": "Lte",
    "in": "In",
    "nin": "NotIn",
    "not_in": "NotIn",
    "contains": "Contains",
}


def _build_filters(filters: dict[str, Any] | None) -> Any | None:
    """Translate a backend-agnostic filter dict into turbopuffer's tuple DSL.

    Single field -> a leaf tuple ``(field, Op, value)``. Multiple fields are
    combined with ``("And", [leaf, leaf, ...])``. Per-field value forms:

      * scalar  ->  ``(field, "Eq", value)``
      * list    ->  ``(field, "In", list)``
      * dict    ->  one leaf per operator, e.g. ``{"gte": x}`` -> ``(field, "Gte", x)``
    """
    if not filters:
        return None

    leaves: list[tuple[Any, ...]] = []
    for field, value in filters.items():
        if isinstance(value, dict):
            for op, operand in value.items():
                key = op.lower()
                if key not in _OP_MAP:
                    raise ValueError(f"Unsupported filter operator: {op!r}")
                leaves.append((field, _OP_MAP[key], operand))
        elif isinstance(value, (list, tuple, set)):
            leaves.append((field, "In", list(value)))
        else:
            leaves.append((field, "Eq", value))

    if not leaves:
        return None
    if len(leaves) == 1:
        return leaves[0]
    return ("And", leaves)


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    """Read one attribute from a turbopuffer row, robust to its access shape.

    Rows expose `.id` and their attributes via either item access (`row[key]`)
    or attribute access (`getattr`); `$dist`-style keys are only reachable by
    item access. Try both before giving up.
    """
    try:
        return row[key]
    except (KeyError, TypeError, IndexError):
        pass
    return getattr(row, key, default)


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Best-effort flatten of a turbopuffer row into a plain dict.

    Prefer the SDK's own mapping view (`dict(row)` / `.model_dump()` /
    `.__dict__`); fall back to probing the known attribute names so we never lose
    metadata fields the demo joins on.
    """
    for attempt in (
        lambda: dict(row) if isinstance(row, dict) else None,
        lambda: row.model_dump() if hasattr(row, "model_dump") else None,
        lambda: dict(row) if hasattr(row, "keys") else None,
    ):
        try:
            data = attempt()
        except Exception:
            data = None
        if isinstance(data, dict) and data:
            return data

    data = getattr(row, "__dict__", None)
    if isinstance(data, dict) and data:
        return dict(data)
    return {}


class TurbopufferBackend:
    """turbopuffer vector + BM25 hybrid backend, fused with client-side RRF."""

    name = "turbopuffer"

    def __init__(
        self,
        namespace: str | None = None,
        api_key: str | None = None,
        region: str | None = None,
        distance_metric: str = "cosine_distance",
    ):
        self.namespace = namespace or os.getenv(
            "TURBOPUFFER_NAMESPACE", "retrieval-bridge-tickets"
        )
        self.api_key = api_key or os.getenv("TURBOPUFFER_API_KEY")
        # Region is part of the host (https://{region}.turbopuffer.com) and is
        # required by the client; default to the same region the docs use.
        self.region = region or os.getenv("TURBOPUFFER_REGION", "gcp-us-central1")
        self.distance_metric = distance_metric

    def _namespace(self):
        """Lazily build the SDK client and return the namespace handle.

        Imported here (not at module top) so the base install never needs the
        turbopuffer SDK; the namespace itself is created implicitly on first write.
        """
        from turbopuffer import Turbopuffer

        if not self.api_key:
            raise RuntimeError(
                "TURBOPUFFER_API_KEY is not set; cannot reach turbopuffer."
            )
        client = Turbopuffer(api_key=self.api_key, region=self.region)
        return client.namespace(self.namespace)

    def upsert(self, docs: list[dict[str, Any]]) -> None:
        """Write documents, declaring the hybrid schema on the way in.

        The vector column is `[dim]f32` with `ann: true` (required for ANN) and
        `text` is flagged `full_text_search` (required for BM25); `dim` is read
        from the first vector. All other attributes are filterable by default and
        need no schema entry.
        """
        if not docs:
            return

        dim = len(docs[0]["vector"])
        schema = {
            "vector": {"type": f"[{dim}]f32", "ann": True},
            "text": {"type": "string", "full_text_search": True},
        }

        ns = self._namespace()
        ns.write(
            upsert_rows=docs,
            distance_metric=self.distance_metric,
            schema=schema,
        )

    def _rows_to_hits(self, rows: list[Any]) -> list[Hit]:
        hits: list[Hit] = []
        for row in rows:
            data = _row_to_dict(row)
            doc_id = data.get("id")
            if doc_id is None:
                doc_id = _row_get(row, "id")

            text = data.get("text")
            if text is None:
                text = _row_get(row, "text", "")

            # turbopuffer returns the distance as $dist (lower is better); we keep
            # it only as a per-modality debug value — RRF re-scores by rank anyway.
            raw = data.get("$dist")
            if raw is None:
                raw = _row_get(row, "$dist", 0.0)

            metadata = {k: v for k, v in data.items() if k not in _RESERVED}

            hits.append(
                Hit(
                    id=str(doc_id),
                    score=float(raw or 0.0),
                    text=text or "",
                    metadata=metadata,
                )
            )
        return hits

    def search(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[Hit]:
        """Hybrid search via one snapshot-isolated multi_query, fused with RRF.

        Sub-query 1 ranks by vector ANN; sub-query 2 ranks by BM25 over `text`.
        Both carry the translated filters and `include_attributes=True` (we need
        the metadata to build `Hit`s and to join downstream). The two ranked lists
        are fused client-side because turbopuffer does no server-side fusion.
        """
        ns = self._namespace()
        tpuf_filters = _build_filters(filters)
        candidates = max(top_k * 4, 20)

        vector_sub: dict[str, Any] = {
            "rank_by": ("vector", "ANN", query_embedding),
            "top_k": candidates,
            "include_attributes": True,
        }
        fts_sub: dict[str, Any] = {
            "rank_by": ("text", "BM25", query_text),
            "top_k": candidates,
            "include_attributes": True,
        }
        if tpuf_filters is not None:
            vector_sub["filters"] = tpuf_filters
            fts_sub["filters"] = tpuf_filters

        response = ns.multi_query(queries=[vector_sub, fts_sub])
        results = response.results

        vector_hits = self._rows_to_hits(results[0].rows)
        fts_hits = self._rows_to_hits(results[1].rows)

        fused = reciprocal_rank_fusion([vector_hits, fts_hits])
        return fused[:top_k]
