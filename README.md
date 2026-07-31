# retrieval-bridge

[![CI](https://github.com/viper9503/retrieval-bridge/actions/workflows/ci.yml/badge.svg)](https://github.com/viper9503/retrieval-bridge/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) ![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue) ![No API keys to run](https://img.shields.io/badge/demo-zero%20API%20keys-brightgreen)

**Hybrid (vector + BM25) retrieval for [PromptQL](https://promptql.io), exposed as a `search_documents` command through a Hasura DDN Python lambda connector, with swappable vector backends — LanceDB (embedded, zero-setup default), [turbopuffer](https://turbopuffer.com) (the serverless deploy target), and pgvector — behind one `VectorBackend` interface.**

One natural-language question → PromptQL generates a deterministic query plan → the plan calls the `search_documents` command → the connector runs a vector ANN sub-query and a BM25 sub-query and fuses them with reciprocal-rank fusion → each hit's declarative `account` relationship joins structured Postgres facts → PromptQL classifies, summarizes, and ranks. The backend behind the command swaps (LanceDB · turbopuffer · pgvector) without changing a line of the PromptQL-facing interface, and the demo runs end to end with **no API keys**.

---

## Why it exists

PromptQL gives you planning, deterministic execution, and reasoning primitives (`classify` / `summarize` / `extract` / `visualize`), but it still needs a retrieval step that narrows a large unstructured corpus down to the handful of documents worth reasoning over. Pure vector search has a blind spot for exact tokens — error codes, ticket ids, SKUs all embed to nearly the same place — while BM25 alone misses paraphrases; hybrid search covers both, and `scripts/bench.py` measures exactly that. This repo packages hybrid retrieval as a first-class DDN command so the planner can call it with zero bespoke plumbing, keeps the vector store swappable behind one interface so no backend is a lock-in, and ships a local zero-key path (fastembed + LanceDB + SQLite) so the whole architecture is runnable on a laptop before anything is deployed.

---

## Architecture

```mermaid
flowchart TD
    Q["Natural-language question"] --> P["PromptQL planner<br/>(generates a Python query plan)"]
    P -->|"search_documents(query, top_k, plan_tier, status)<br/>ddn/metadata/search_documents.hml"| C["Python lambda connector<br/>ddn/connector/search/functions.py"]
    C --> B["RetrievalBridge.search_documents<br/>retrieval_bridge/search.py<br/>(embeds the query once)"]
    B --> VB{"VectorBackend.search<br/>retrieval_bridge/backends/"}
    VB --> V["vector ANN sub-query<br/>(semantic similarity)"]
    VB --> T["BM25 full-text sub-query<br/>(exact tokens: ERR_DIM_384, ids)"]
    V --> F["reciprocal_rank_fusion<br/>retrieval_bridge/fusion.py"]
    T --> F
    F -->|"top_k SearchHit rows"| RT["PromptQL runtime<br/>classify · summarize · rank"]
    RT -->|"SearchHit.account relationship<br/>id → Accounts.ticket_id<br/>ddn/metadata/relationship_searchhit_account.hml"| A[("Accounts model — Postgres<br/>plan_tier · monthly_revenue · region")]
    RT --> ANS["Ranked, enriched answer"]

    subgraph stores ["Swappable stores — RETRIEVAL_BRIDGE_BACKEND"]
        S1[("LanceDB<br/>embedded · default")]
        S2[("turbopuffer<br/>serverless deploy target")]
        S3[("pgvector<br/>Postgres-native")]
    end
    VB -.-> S1
    VB -.-> S2
    VB -.-> S3
```

Every box maps to real code. The command body is [`RetrievalBridge.search_documents`](retrieval_bridge/search.py); the deployed connector ([`ddn/connector/search/functions.py`](ddn/connector/search/functions.py)) is a thin adapter around it, so the local demo and the deployed connector execute the identical retrieval path. The `Hit` / `SearchHit` contracts live in [`retrieval_bridge/types.py`](retrieval_bridge/types.py) and [`ddn/metadata/SearchHit.hml`](ddn/metadata/SearchHit.hml).

<details>
<summary>Same flow as ASCII (for non-Mermaid viewers)</summary>

```
User (natural language)
        │
        ▼
   PromptQL planner ──────────► generates a Python query plan
        │
        ▼
  command: search_documents(query, top_k, plan_tier, status)
        │        ddn/metadata/search_documents.hml
        ▼
  Python lambda connector          ddn/connector/search/functions.py
        │  = RetrievalBridge.search_documents   retrieval_bridge/search.py
        ▼
  VectorBackend.search  (one interface, three stores)
     ├── vector ANN sub-query   (semantic)
     └── BM25 sub-query         (exact tokens)
              │
              ▼
  reciprocal_rank_fusion           retrieval_bridge/fusion.py
              │
              ▼
  top_k SearchHit rows ──► PromptQL runtime (classify · summarize · rank)
              │
              └── SearchHit.account relationship (id → Accounts.ticket_id)
                            │
                            ▼
                  Postgres Accounts model (plan_tier · monthly_revenue)
```

</details>

---

## Design decisions

### Reciprocal-rank fusion, client-side

Cosine distances and BM25 scores live on incomparable scales, so merging the two ranked lists by score would require per-backend (and per-corpus) normalization tuning. [`reciprocal_rank_fusion`](retrieval_bridge/fusion.py) sidesteps that entirely by fusing on **rank alone** — `score(d) = Σ w_L / (k + rank_L(d))` with the canonical `k = 60` — which makes it score-scale-free: nothing to calibrate when the store, embedder, or corpus changes, and optional per-list weights remain available to upweight BM25 for token-heavy queries. Fusing client-side is also what makes "hybrid" mean the same thing everywhere: turbopuffer has **no server-side rank fusion** (its documented hybrid recipe is one snapshot-isolated `multi_query` with a vector sub-query and a BM25 sub-query, fused by the caller), so LanceDB and pgvector mirror that shape and all three reuse this one function. Its ordering, dedup, weighting, and damping properties are pinned by [`tests/test_fusion.py`](tests/test_fusion.py).

### One `VectorBackend` protocol, three stores

[`retrieval_bridge/backends/base.py`](retrieval_bridge/backends/base.py) defines a two-method `Protocol`: `upsert(docs)` and `search(query_embedding, query_text, top_k, filters) -> list[Hit]`. `search` deliberately receives both the query embedding (for the vector sub-query) and the raw query text (for BM25), and every backend returns the same `Hit` model — so the PromptQL-facing command signature never changes when the store does. [`get_backend()`](retrieval_bridge/backends/__init__.py) selects the implementation from `RETRIEVAL_BRIDGE_BACKEND` (default `lancedb`) and imports SDKs lazily, so the base install needs only LanceDB. Filters cross the seam as one backend-agnostic dict (equality, membership, and `gte`-style operator forms) that each backend renders into its native DSL: a LanceDB SQL predicate, turbopuffer's tuple DSL (`("plan_tier", "Eq", ...)`), or a parameterized Postgres `WHERE` over the JSONB metadata column. Swapping stores changes only where the two sub-queries execute.

### A DDN lambda connector, not a standalone API

Retrieval is only useful to PromptQL if the planner can discover it as a typed tool, and DDN's lambda connector is the seam that already provides this. In [`ddn/connector/search/functions.py`](ddn/connector/search/functions.py), `@connector.register_query` (via `ndc_sdk_python`) turns `search_documents` into a DDN Command: the function's docstring seeds the planner-facing command description (hand-tuned in [`ddn/metadata/search_documents.hml`](ddn/metadata/search_documents.hml) — that prose, not the code, drives tool selection), and the typed pydantic return becomes the command's object type. `SearchHit` is deliberately a **named** type with flat scalar fields rather than `dict`/`Any`: DDN infers the NDC object type from the type hints, the planner filters and orders on those columns, and only a named type with a concrete `id` field lets a relationship attach to the result. A standalone HTTP API would need transport, auth, discovery, and a client shim as bespoke plumbing — and still could not participate in DDN relationships.

### Declarative relationship for the structured join

A retrieved ticket becomes decision-ready only when joined to account facts (plan tier, monthly revenue), and that join is metadata, not code: [`ddn/metadata/relationship_searchhit_account.hml`](ddn/metadata/relationship_searchhit_account.hml) maps `SearchHit.id → Accounts.ticket_id` (`relationshipType: Object`), so a plan or GraphQL query selects `account { plan_tier monthly_revenue }` per hit and the engine resolves it. The retrieval layer stays ignorant of the warehouse schema, and the join evolves with the metadata rather than with connector releases. Two hard-won details are documented in the file itself: it must be **hand-authored**, because `ddn relationship add` only generates relationships from Postgres foreign keys between Models (this source is a command return type), and `sourceType` must be the **return type name** (`SearchHit`), never the command name. The local demo mirrors the same resolution with [`StructuredStore.get_by_ids`](retrieval_bridge/structured.py) over SQLite.

---

## How to run

### Quickstart (zero API keys, ~2 minutes)

Requires Python 3.10+. The default path uses local embeddings (fastembed, BGE-small-en-v1.5, 384-dim) and an embedded LanceDB store — nothing to sign up for.

```bash
git clone https://github.com/viper9503/retrieval-bridge.git
cd retrieval-bridge
python -m venv .venv && source .venv/bin/activate      # or: uv venv

make install       # pip install -e .  (base = local embeddings + LanceDB)
make all           # corpus + seed + demo, end to end
```

Or the individual steps (each `make` target wraps the script next to it):

```bash
python data/generate_corpus.py     # make corpus — 160 synthetic support tickets (deterministic)
python scripts/seed.py             # make seed   — embed + index (first run downloads BGE once, ~130 MB)
python scripts/demo.py             # make demo   — the end-to-end query plan
python scripts/bench.py            # make bench  — hybrid vs pure-vector on exact tokens
python -m pytest -q                # make test   — unit tests (fusion, backends, primitives)
make clean                         # remove .lancedb / .structured.db local stores
```

`demo.py` executes the full plan a PromptQL prompt would generate — retrieve → classify root cause → summarize the fix → join account facts → rank by plan tier and recency — and prints each step so the division of labor is visible. The classify/summarize steps use deterministic stand-ins ([`retrieval_bridge/demo_primitives.py`](retrieval_bridge/demo_primitives.py)) for the real `executor.*` primitives, so the demo stays offline and reproducible.

### Try your own questions

The demo isn't tied to one incident — run any prompt, or pick from 10 built-in examples:

```bash
python scripts/demo.py --list                       # list the built-in example incidents
python scripts/demo.py --example oom                # OOM/crash incident
python scripts/demo.py --example dim                # embedding dimension mismatch (ranks enterprise first)
python scripts/demo.py --query "users can't log in, 401 after SSO"
python scripts/demo.py -q "..." --top-k 5 --status resolved --plan-tier enterprise
```

Or explore retrieval interactively — just the `search_documents` step (vector + BM25 + RRF), no plan:

```bash
python scripts/ask.py                               # REPL: type queries, see ranked hits
python scripts/ask.py "ERR_TLS_526 certificate"     # one-shot
python scripts/ask.py "throttled during big uploads" --plan-tier enterprise
```

### Run it against real turbopuffer (optional)

```bash
pip install -e ".[turbopuffer]"
export TURBOPUFFER_API_KEY=...            # see docs/turbopuffer-runbook.md
export TURBOPUFFER_REGION=gcp-us-central1
RETRIEVAL_BRIDGE_BACKEND=turbopuffer python scripts/seed.py
RETRIEVAL_BRIDGE_BACKEND=turbopuffer python scripts/demo.py
```

The plan and the command signature are identical — only where the two sub-queries run changes.

### Configuration

Copy `.env.example` to `.env` and fill in only what you need; the default demo needs nothing.

| Variable | Default | Used when |
|---|---|---|
| `RETRIEVAL_BRIDGE_BACKEND` | `lancedb` | Backend selection: `lancedb` \| `turbopuffer` \| `pgvector` |
| `RETRIEVAL_BRIDGE_EMBEDDER` | `local` | Embedder selection: `local` \| `openai` \| `voyage` \| `cohere` |
| `RETRIEVAL_BRIDGE_LANCEDB_PATH` | `./.lancedb` | LanceDB data directory |
| `TURBOPUFFER_API_KEY` | — | Backend = `turbopuffer` (no free tier; lowest plan is $64/mo, 30-day refund window) |
| `TURBOPUFFER_REGION` | `gcp-us-central1` | Backend = `turbopuffer`; must match your key's region (the region is part of the API host) |
| `TURBOPUFFER_NAMESPACE` | `retrieval-bridge-tickets` | Backend = `turbopuffer` |
| `PGVECTOR_DSN` | `postgresql://postgres:postgres@localhost:5432/retrieval_bridge` | Backend = `pgvector` (needs Postgres with the `vector` extension, e.g. `docker run -e POSTGRES_PASSWORD=postgres -p 5432:5432 pgvector/pgvector:pg16`) |
| `OPENAI_API_KEY` / `VOYAGE_API_KEY` / `CO_API_KEY` | — | Cloud embedders (each an optional extra: `.[openai]` / `.[voyage]` / `.[cohere]`) |
| `ANTHROPIC_API_KEY` | — | Deployed DDN only — read by `ddn/globals/promptql-config.hml` for the PromptQL planner/primitives LLM. The local demo always uses the deterministic stand-ins. |

---

## Swappable backends

| Backend | Install | Role | Hybrid |
|---|---|---|---|
| **LanceDB** (default) | base | Zero-infra, embedded, object-storage-native — the open-source turbopuffer analog and clone-and-run default | vector ANN + native FTS + client-side RRF |
| **turbopuffer** | `.[turbopuffer]` | Serverless hybrid search over object storage at scale — the deploy target this repo is pitched around | vector ANN + BM25 `multi_query` + client-side RRF |
| **pgvector** | `.[pgvector]` | Lowest friction inside DDN (native Postgres connector); corpus and `accounts` can share one database | vector `<=>` + `tsvector`/`ts_rank` + client-side RRF |

All three implement the same [`VectorBackend`](retrieval_bridge/backends/base.py) protocol and reuse the same [`reciprocal_rank_fusion`](retrieval_bridge/fusion.py), so "hybrid" means the same thing everywhere.

---

## Repo layout

```
retrieval_bridge/        # the backend-agnostic core (shared by demo AND connector)
  types.py               #   Hit / Document contracts
  fusion.py              #   reciprocal-rank fusion (the client-side hybrid step)
  embedders/             #   local BGE (default) + OpenAI/Voyage/Cohere (asymmetric-aware)
  backends/              #   lancedb (default) + turbopuffer + pgvector
  search.py              #   RetrievalBridge.search_documents  <- the command body
  structured.py          #   SQLite structured store (the DDN Postgres-join analog)
  demo_primitives.py     #   offline stand-ins for executor.classify/summarize
data/generate_corpus.py  # synthetic support tickets (semantic + exact tokens)
scripts/                 # seed.py · demo.py · ask.py · bench.py
ddn/                     # the deployable DDN connector + HML metadata + runbook
tests/                   # fusion, LanceDB backend, demo-primitive unit tests
docs/                    # architecture, pitch, query-plan walkthrough, runbooks
```

---

## Going live (turbopuffer + PromptQL)

Two runbooks, for going from the local demo to a deployed stack:

- **[docs/turbopuffer-runbook.md](docs/turbopuffer-runbook.md)** — sign up, get a key, choose a region, seed a real namespace. Note turbopuffer has **no free tier** (lowest plan is $64/mo minimum, with a 30-day refund window) — which is exactly why the local default exists: the demo is free and the upgrade is one env var.
- **[docs/ddn-runbook.md](docs/ddn-runbook.md)** — install the DDN CLI, drop in the lambda connector, register the `search_documents` command, add the Postgres model + relationship, and point your existing PromptQL project at it.
- **[docs/query-plan-walkthrough.md](docs/query-plan-walkthrough.md)** — the demo prompt and the query plan PromptQL is expected to generate, annotated.

---

## A note on accuracy

Every external API in this repo was verified against live documentation (June 2026), not training memory — which caught several breaking drifts (turbopuffer's v2 API + region-scoped hosts, the `ndc_sdk_python` connector imports, the four PromptQL primitives, Voyage `voyage-3` deprecation). See [docs/verified-facts.md](docs/verified-facts.md) for the citations behind the build.

## Docs

- [docs/pitch.md](docs/pitch.md) — the extended pitch (why this combination matters)
- [docs/architecture.md](docs/architecture.md) — architecture deep dive
- [docs/query-plan-walkthrough.md](docs/query-plan-walkthrough.md) — the demo prompt + the PromptQL plan it should generate
- [docs/turbopuffer-runbook.md](docs/turbopuffer-runbook.md) — go live on real turbopuffer
- [docs/ddn-runbook.md](docs/ddn-runbook.md) — wire the connector + relationship into DDN/PromptQL
- [docs/verified-facts.md](docs/verified-facts.md) — every external API, verified against live docs
- [docs/recruiter-blurb.md](docs/recruiter-blurb.md) — copy-paste blurb + a 60-second demo script

## License

MIT — see [LICENSE](LICENSE).
