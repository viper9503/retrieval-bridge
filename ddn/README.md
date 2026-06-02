# `ddn/` — the deployable Hasura DDN connector + metadata

This directory is the **deployed** half of retrieval-bridge: the Python lambda
connector plus the Hasura DDN metadata (HML) that exposes hybrid retrieval to
PromptQL as a first-class tool. The `retrieval_bridge/` package in the repo root
is the backend-agnostic core shared by both the local demo *and* this connector —
the same `RetrievalBridge.search_documents` runs in each.

> **New to DDN setup?** Follow **[../docs/ddn-runbook.md](../docs/ddn-runbook.md)**
> for the step-by-step: install the CLI, add the connector, register the command,
> add the Postgres model + relationship, and point PromptQL at it. This README is
> the map; the runbook is the walkthrough.

---

## How the pieces connect

```
User (natural language)
        │
        ▼
  PromptQL planner ── steered by ──► globals/promptql-config.hml
        │                              (systemInstructions: "use search_documents,
        │                               follow the account relationship")
        ▼
  command: search_documents(query, top_k, plan_tier, status)
        │   metadata/search_documents.hml      (kind: Command)
        ▼
  Python lambda connector  ── connector/search/functions.py
        │   = RetrievalBridge.search_documents  (the SAME core code)
        ▼
  returns [SearchHit!]!     metadata/SearchHit.hml   (named ObjectType, has `id`)
        │
        └── each SearchHit.account ──► metadata/relationship_searchhit_account.hml
                                        (hand-authored join: id → ticket_id)
                                              │
                                              ▼
                                    Accounts model   metadata/accounts.hml
                                    (Postgres table  metadata/accounts.sql)
```

The retrieval **backend** behind the connector (turbopuffer · LanceDB · pgvector)
is swappable without changing any of this metadata — the command signature is the
stable interface.

---

## Files

| Path | Kind | Authored how |
|---|---|---|
| `metadata/search_documents.hml` | `Command` v1 | Draft generated on introspect; the **description is hand-tuned** (it drives PromptQL tool selection). |
| `metadata/SearchHit.hml` | `ObjectType` v1 | **Normally generated** by `ddn connector-link add-resources` from the connector return type. Committed here as reference. |
| `metadata/accounts.hml` | `ObjectType` + `Model` v1 | **Normally generated** by `ddn model add` from the Postgres table. |
| `metadata/accounts.sql` | Postgres DDL | The table the Postgres connector introspects; load rows from `data/tickets.jsonl` `account` objects. |
| `metadata/relationship_searchhit_account.hml` | `Relationship` v1 | **Hand-authored** — `ddn relationship add` only does Postgres FK relationships, not command-return-type → model joins. |
| `globals/promptql-config.hml` | `PromptQlConfig` v2 | The Anthropic LLM config + `systemInstructions` that steer the planner. |
| `connector/search/` | lambda connector | The Python `ndc-lambda`/`ndc_sdk_python` connector whose `search_documents` function is the command body. See the DDN runbook. |

---

## The three things that make the join work (gotchas)

1. **The return type must be a NAMED ObjectType with an `id` field.** The lambda
   returns a typed object → DDN names it `SearchHit`. Relationships attach to a
   *type*, and the mapping needs a concrete source field (`id`).

2. **The relationship is hand-written, and `sourceType` is the RETURN TYPE NAME.**
   `relationship_searchhit_account.hml` sets `sourceType: SearchHit` (the type
   `search_documents` returns) — **never** the command name. `relationshipType:
   Object` is required for the single-row model target.

3. **The `description` fields are the tool spec.** PromptQL chooses tools from
   the command/relationship `description` and from `promptql-config.hml`
   `systemInstructions` — not from Python docstrings (though the connector
   docstring seeds the command description on introspect). If PromptQL isn't
   reaching for retrieval, tune that prose, not the code.

> ⚠️ Model identifiers in `promptql-config.hml` are **point-in-time** Anthropic
> snapshots and will drift — replace with a current model id before deploying.

---

## Local equivalents (so you can see it before deploying)

Everything above has a zero-key local analog you can run today (`python
scripts/demo.py`):

| Deployed (this dir) | Local analog |
|---|---|
| `search_documents` command | `retrieval_bridge/search.py` (same function) |
| `SearchHit` return type | `retrieval_bridge/types.py` `Hit` |
| `executor.classify` / `executor.summarize` | `retrieval_bridge/demo_primitives.py` |
| `SearchHit.account` DDN relationship | `retrieval_bridge/structured.py` `get_by_ids` (SQLite) |
| Postgres `accounts` model | the SQLite `accounts` table (same columns) |

See the repo-root [README.md](../README.md) for the full pitch and quickstart.
