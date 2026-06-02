"""pgvector backend — the DDN-native-join story.

This is the "you already run Postgres" answer to the swappable-backend pitch.
It mirrors turbopuffer's hybrid recipe with two stock Postgres primitives:

  * a `vector(DIM)` column queried with the cosine-distance operator `<=>`
    (the pgvector ANN sub-query), and
  * `to_tsvector('english', text) @@ plainto_tsquery(...)` scored by `ts_rank`
    (the full-text / BM25-ish sub-query).

turbopuffer has **no server-side rank fusion**, and neither does this backend —
the two ranked lists are fused client-side with the shared
`reciprocal_rank_fusion`, exactly as on turbopuffer and LanceDB. The PromptQL
`search_documents` command therefore runs unchanged; only *where* the two
sub-queries execute moves. The bonus here is locality: because the corpus and
the Accounts table can live in the same Postgres, the DDN hit.id -> Accounts
relationship can be a native SQL join rather than a cross-source one.

Unlike LanceDB this backend needs a running Postgres with the `vector`
extension. One-liner:

    docker run -e POSTGRES_PASSWORD=postgres -p 5432:5432 pgvector/pgvector:pg16

The psycopg3 and pgvector SDKs are imported lazily (inside `_connect`) so the
base install does not require them.
"""

from __future__ import annotations

import json
import os
from typing import Any

from ..fusion import reciprocal_rank_fusion
from ..types import Hit

_RESERVED = {"id", "text", "vector"}

_DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5432/retrieval_bridge"

# Backend-agnostic operator dict -> SQL comparison operator.
_OPS = {"eq": "=", "neq": "!=", "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}


class PgvectorBackend:
    """Postgres + pgvector hybrid backend (vector `<=>` + FTS), fused with RRF."""

    name = "pgvector"

    def __init__(
        self,
        dsn: str | None = None,
        table: str = "documents",
    ):
        self.dsn = dsn or os.getenv("PGVECTOR_DSN", _DEFAULT_DSN)
        self.table = table
        self._dim: int | None = None

    # -- connection -----------------------------------------------------------

    def _connect(self):
        """Open a psycopg3 connection with the vector extension + adapter ready.

        Imported lazily so `import retrieval_bridge` never requires psycopg.
        """
        import psycopg
        from pgvector.psycopg import register_vector

        conn = psycopg.connect(self.dsn, autocommit=True)
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        return conn

    # -- filter translation ---------------------------------------------------

    def _build_where(self, filters: dict[str, Any] | None) -> tuple[str, list[Any]]:
        """Translate a backend-agnostic filter dict into a parameterized WHERE.

        Returns the SQL fragment (without the leading "WHERE") and the ordered
        parameter list. Supports scalar equality ({"status": "resolved"}),
        membership ({"plan_tier": ["scale", "enterprise"]} -> = ANY(...)), and
        operator dicts ({"created_at": {"gte": "2025-01-01"}}). Values are read
        out of the JSONB metadata column as text via `metadata->>'field'`, which
        is correct for the string-valued attrs and fine for ISO dates that sort
        lexicographically.
        """
        if not filters:
            return "", []

        clauses: list[str] = []
        params: list[Any] = []
        for field, value in filters.items():
            col = "metadata->>%s"
            if isinstance(value, dict):
                for op, operand in value.items():
                    if op not in _OPS:
                        raise ValueError(f"Unsupported filter operator: {op!r}")
                    # JSONB ->> yields text, so numeric comparisons must cast to
                    # ::numeric or they compare lexicographically (10 < 9). Dates
                    # are ISO strings and sort correctly as text, so only cast
                    # genuine numbers (bools excluded).
                    if isinstance(operand, (int, float)) and not isinstance(operand, bool):
                        clauses.append(f"((metadata->>%s)::numeric {_OPS[op]} %s)")
                        params.extend([field, operand])
                    else:
                        clauses.append(f"({col} {_OPS[op]} %s)")
                        params.extend([field, _scalar(operand)])
            elif isinstance(value, (list, tuple, set)):
                clauses.append(f"({col} = ANY(%s))")
                params.extend([field, [_scalar(v) for v in value]])
            elif value is None:
                clauses.append("(metadata->>%s IS NULL)")
                params.append(field)
            else:
                clauses.append(f"({col} = %s)")
                params.extend([field, _scalar(value)])
        return " AND ".join(clauses), params

    # -- write ----------------------------------------------------------------

    def upsert(self, docs: list[dict[str, Any]]) -> None:
        """Index documents into the `documents` table (INSERT ... ON CONFLICT).

        The vector column dimension is inferred from the first doc's vector, so
        the table is created on first write. Every key other than id/text/vector
        is stored in a single JSONB `metadata` column, matching the filter
        semantics in `_build_where`.
        """
        if not docs:
            return

        first_vec = docs[0].get("vector")
        if first_vec is None:
            raise ValueError("each doc must carry a 'vector'; embed before upsert")
        self._dim = len(first_vec)

        conn = self._connect()
        try:
            self._ensure_schema(conn, self._dim)

            sql = (
                f"INSERT INTO {self.table} (id, text, metadata, embedding) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET "
                "text = EXCLUDED.text, "
                "metadata = EXCLUDED.metadata, "
                "embedding = EXCLUDED.embedding"
            )
            with conn.cursor() as cur:
                for doc in docs:
                    metadata = {
                        k: v for k, v in doc.items() if k not in _RESERVED
                    }
                    cur.execute(
                        sql,
                        (
                            str(doc["id"]),
                            doc.get("text", ""),
                            json.dumps(metadata),
                            list(doc["vector"]),
                        ),
                    )

            self._ensure_indexes(conn)
        finally:
            conn.close()

    def _ensure_schema(self, conn, dim: int) -> None:
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {self.table} ("
            "id TEXT PRIMARY KEY, "
            "text TEXT, "
            "metadata JSONB, "
            f"embedding vector({dim})"
            ")"
        )

    def _ensure_indexes(self, conn) -> None:
        """Best-effort index creation; never fatal to an upsert.

        A GIN index over `to_tsvector('english', text)` accelerates the FTS
        sub-query, and an IVFFlat (cosine) index accelerates the ANN sub-query.
        Both are wrapped so a missing/old pgvector or an empty table can't break
        seeding — the queries are correct without the indexes, just slower.
        """
        try:
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table}_fts_idx "
                f"ON {self.table} USING GIN (to_tsvector('english', text))"
            )
        except Exception:
            pass
        try:
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table}_embedding_idx "
                f"ON {self.table} USING ivfflat (embedding vector_cosine_ops) "
                "WITH (lists = 100)"
            )
        except Exception:
            pass

    # -- read -----------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[Hit]:
        """Hybrid search: pgvector ANN + Postgres FTS, fused client-side by RRF."""
        where_sql, where_params = self._build_where(filters)
        where_clause = f" AND {where_sql}" if where_sql else ""
        candidates = max(top_k * 4, 20)

        conn = self._connect()
        try:
            # Guard against a cold search before any upsert created the table.
            with conn.cursor() as cur:
                cur.execute("SELECT to_regclass(%s)", [self.table])
                if cur.fetchone()[0] is None:
                    return []

            # --- vector ANN sub-query (cosine: 1 - distance => similarity) ---
            vec = list(query_embedding)
            vector_sql = (
                "SELECT id, text, metadata, "
                "1 - (embedding <=> %s::vector) AS score "
                f"FROM {self.table} "
                "WHERE TRUE"
                + where_clause
                + " ORDER BY embedding <=> %s::vector LIMIT %s"
            )
            with conn.cursor() as cur:
                cur.execute(
                    vector_sql,
                    [vec, *where_params, vec, candidates],
                )
                vector_hits = _rows_to_hits(cur.fetchall())

            # --- full-text (BM25-ish) sub-query, ranked by ts_rank ---
            fts_sql = (
                "SELECT id, text, metadata, "
                "ts_rank(to_tsvector('english', text), "
                "plainto_tsquery('english', %s)) AS score "
                f"FROM {self.table} "
                "WHERE to_tsvector('english', text) @@ "
                "plainto_tsquery('english', %s)"
                + where_clause
                + " ORDER BY score DESC LIMIT %s"
            )
            with conn.cursor() as cur:
                cur.execute(
                    fts_sql,
                    [query_text, query_text, *where_params, candidates],
                )
                fts_hits = _rows_to_hits(cur.fetchall())
        finally:
            conn.close()

        fused = reciprocal_rank_fusion([vector_hits, fts_hits])
        return fused[:top_k]


def _scalar(v: Any) -> str:
    """Render a filter operand the way `metadata->>'field'` returns it: as text.

    JSONB `->>` always yields text, so comparisons must be against text. Bools
    are lowercased to match JSON's `true`/`false`; everything else is stringified.
    """
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _rows_to_hits(rows: list[tuple[Any, ...]]) -> list[Hit]:
    """Map (id, text, metadata, score) rows into `Hit`s.

    The raw per-modality score is carried for debugging; RRF re-scores anyway.
    `metadata` is the JSONB column, already decoded to a dict by psycopg.
    """
    hits: list[Hit] = []
    for row in rows:
        doc_id, text, metadata, score = row
        if metadata is None:
            metadata = {}
        elif isinstance(metadata, str):  # defensive: if JSONB came back as text
            metadata = json.loads(metadata)
        hits.append(
            Hit(
                id=str(doc_id),
                score=float(score or 0.0),
                text=text or "",
                metadata=dict(metadata),
            )
        )
    return hits
