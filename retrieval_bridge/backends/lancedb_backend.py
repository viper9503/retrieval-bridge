"""LanceDB backend — the zero-infra default.

LanceDB is an embedded, object-storage-native vector database (the closest
open-source analog to turbopuffer's architecture), which makes it the ideal
clone-and-run default *and* the honest "no vendor lock-in" talking point: the
exact same `search_documents` command runs unchanged on it.

It mirrors turbopuffer's hybrid recipe faithfully — a vector ANN sub-query and a
BM25 full-text sub-query, fused client-side with the shared
`reciprocal_rank_fusion`. Nothing about hybrid search is delegated to a
proprietary endpoint, so swapping to turbopuffer (or pgvector) changes only where
the two sub-queries run.
"""

from __future__ import annotations

import os
from typing import Any

from ..fusion import reciprocal_rank_fusion
from ..types import Hit

_RESERVED = {"vector", "_distance", "_score", "_relevance_score", "_rowid"}


def _build_where(filters: dict[str, Any] | None) -> str | None:
    """Translate a backend-agnostic filter dict into a LanceDB SQL predicate.

    Supports: scalar equality ({"status": "resolved"}), membership
    ({"plan_tier": ["scale", "enterprise"]}), and operator dicts
    ({"created_at": {"gte": "2025-01-01"}}). Operators: eq, neq, gt, gte, lt, lte.
    """
    if not filters:
        return None

    def lit(v: Any) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        return "'" + str(v).replace("'", "''") + "'"

    ops = {"eq": "=", "neq": "!=", "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}
    clauses: list[str] = []
    for field, value in filters.items():
        if isinstance(value, dict):
            for op, operand in value.items():
                if op not in ops:
                    raise ValueError(f"Unsupported filter operator: {op!r}")
                clauses.append(f"{field} {ops[op]} {lit(operand)}")
        elif isinstance(value, (list, tuple, set)):
            joined = ", ".join(lit(v) for v in value)
            clauses.append(f"{field} IN ({joined})")
        else:
            clauses.append(f"{field} = {lit(value)}")
    return " AND ".join(clauses) if clauses else None


class LanceDBBackend:
    """Embedded vector + BM25 hybrid backend, fused with client-side RRF."""

    name = "lancedb"

    def __init__(
        self,
        uri: str | None = None,
        table: str = "tickets",
        distance: str = "cosine",
    ):
        self.uri = uri or os.getenv("RETRIEVAL_BRIDGE_LANCEDB_PATH", "./.lancedb")
        self.table = table
        self.distance = distance

    def _connect(self):
        import lancedb

        return lancedb.connect(self.uri)

    def upsert(self, docs: list[dict[str, Any]]) -> None:
        """(Re)build the table from `docs` and create the full-text index.

        For a demo corpus this overwrites the table in one shot; LanceDB infers
        the fixed-size vector column from the data.
        """
        if not docs:
            return
        db = self._connect()
        tbl = db.create_table(self.table, data=docs, mode="overwrite")
        # Native (tantivy-free) full-text index over the ticket body -> BM25.
        tbl.create_fts_index("text", replace=True, use_tantivy=False)

    def _open(self):
        db = self._connect()
        return db.open_table(self.table)

    def _rows_to_hits(self, rows: list[dict[str, Any]]) -> list[Hit]:
        hits: list[Hit] = []
        for row in rows:
            metadata = {k: v for k, v in row.items() if k not in _RESERVED and k not in ("id", "text")}
            # carry a per-modality raw score for debugging; RRF re-scores anyway.
            raw = row.get("_relevance_score", row.get("_score", row.get("_distance", 0.0)))
            hits.append(
                Hit(id=str(row["id"]), score=float(raw or 0.0), text=row.get("text", ""), metadata=metadata)
            )
        return hits

    def search(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[Hit]:
        tbl = self._open()
        where = _build_where(filters)
        candidates = max(top_k * 4, 20)

        # --- vector ANN sub-query ---
        vq = tbl.search(query_embedding, vector_column_name="vector").limit(candidates)
        if where:
            vq = vq.where(where, prefilter=True)
        vector_hits = self._rows_to_hits(vq.to_list())

        # --- BM25 full-text sub-query ---
        fts_hits: list[Hit] = []
        try:
            fq = tbl.search(query_text, query_type="fts").limit(candidates)
            if where:
                fq = fq.where(where)
            fts_hits = self._rows_to_hits(fq.to_list())
        except Exception:
            # If the FTS index is unavailable, degrade to vector-only rather than
            # crash the demo. Hybrid is the headline, so this should not fire.
            fts_hits = []

        fused = reciprocal_rank_fusion([vector_hits, fts_hits])
        return fused[:top_k]
