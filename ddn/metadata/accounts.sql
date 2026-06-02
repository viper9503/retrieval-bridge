-- ────────────────────────────────────────────────────────────────────────────
-- accounts — Postgres DDL for the structured-facts table behind the Accounts Model
-- ────────────────────────────────────────────────────────────────────────────
--
-- This is the table the Hasura DDN Postgres connector introspects to produce
-- accounts.hml (via `ddn model add`). It is the join target for retrieved
-- SearchHits: SearchHit.id  →  accounts.ticket_id (see
-- relationship_searchhit_account.hml).
--
-- The column set matches retrieval_bridge/structured.py (the SQLite analog used
-- by the local demo) exactly, so the deployed Postgres join and the local SQLite
-- join behave identically.
--
-- USAGE
--   1. Create a Postgres database and run this file:
--          createdb retrieval_bridge
--          psql retrieval_bridge -f ddn/metadata/accounts.sql
--   2. Load the per-ticket account facts. They live in data/tickets.jsonl under
--      each line's "account" object — the SAME rows scripts/seed.py upserts into
--      SQLite. A quick one-liner to load them (psql + jq):
--
--          jq -c '.account' data/tickets.jsonl \
--            | jq -r '[.ticket_id,.project_id,.project_name,.plan_tier,
--                      .monthly_revenue,.account_region,.seat_count]
--                     | @csv' \
--            | psql retrieval_bridge -c \
--                "COPY accounts (ticket_id, project_id, project_name, plan_tier,
--                                monthly_revenue, account_region, seat_count)
--                 FROM STDIN WITH (FORMAT csv)"
--
--      (Any loader works — the point is: one row per ticket, from the "account"
--      object in data/tickets.jsonl.)
--   3. Point the DDN Postgres connector at this database and run
--      `ddn model add <pg-connector-link> accounts`.
-- ────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS accounts (
    ticket_id        text             PRIMARY KEY,     -- join key: SearchHit.id → here
    project_id       text             NOT NULL,
    project_name     text             NOT NULL,
    plan_tier        text             NOT NULL,         -- free | launch | scale | enterprise
    monthly_revenue  double precision NOT NULL,         -- account MRR (USD)
    account_region   text             NOT NULL,
    seat_count       integer          NOT NULL
);

-- Helps filtering/ranking hits by plan tier (the demo ranks scale/enterprise up).
CREATE INDEX IF NOT EXISTS accounts_plan_tier_idx ON accounts (plan_tier);
