"""Structured store — the standalone analog of the DDN Postgres join.

In the deployed architecture, a retrieved ticket id joins to its structured row
(project, plan tier, monthly revenue) through a **declarative DDN relationship**
(see ddn/metadata/relationship_searchhit_account.hml): no join code, the engine
resolves `searchHit.account` from the hit's id.

There is no Postgres in a clone-and-run demo, so we use SQLite to play the same
role: `get_by_ids` is exactly the lookup the DDN relationship performs for you.
The demo prints this mapping explicitly so the equivalence is obvious.
"""

from __future__ import annotations

import sqlite3
from typing import Any

# The structured "facts" table. In DDN this is a Postgres table exposed as a
# Model; here it is a SQLite table queried by id.
SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    ticket_id       TEXT PRIMARY KEY,
    project_id      TEXT NOT NULL,
    project_name    TEXT NOT NULL,
    plan_tier       TEXT NOT NULL,
    monthly_revenue REAL NOT NULL,
    account_region  TEXT NOT NULL,
    seat_count      INTEGER NOT NULL
);
"""

COLUMNS = [
    "ticket_id",
    "project_id",
    "project_name",
    "plan_tier",
    "monthly_revenue",
    "account_region",
    "seat_count",
]


class StructuredStore:
    """SQLite-backed structured facts, keyed by ticket id."""

    def __init__(self, path: str = "./.structured.db"):
        self.path = path

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    def upsert(self, rows: list[dict[str, Any]]) -> None:
        self.init_schema()
        placeholders = ", ".join("?" for _ in COLUMNS)
        sql = (
            f"INSERT OR REPLACE INTO accounts ({', '.join(COLUMNS)}) "
            f"VALUES ({placeholders})"
        )
        with self._conn() as conn:
            conn.executemany(sql, [[row[c] for c in COLUMNS] for row in rows])

    def get_by_ids(self, ticket_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Return {ticket_id: structured_row} for the given ids.

        This is the operation the DDN relationship performs implicitly when you
        request `searchHit { account { plan_tier monthly_revenue } }`.
        """
        if not ticket_ids:
            return {}
        marks = ", ".join("?" for _ in ticket_ids)
        with self._conn() as conn:
            cur = conn.execute(
                f"SELECT * FROM accounts WHERE ticket_id IN ({marks})", list(ticket_ids)
            )
            return {row["ticket_id"]: dict(row) for row in cur.fetchall()}
