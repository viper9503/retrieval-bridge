# `search` connector — DDN Python lambda

This is the deploy-side wrapper around `RetrievalBridge.search_documents`. It
exposes one PromptQL Command, `search_documents`, backed by turbopuffer-style
hybrid retrieval (vector ANN + BM25, fused client-side with reciprocal-rank
fusion). The backend is swappable behind the bridge, so DDN never learns which
vector store answered — turbopuffer for the headline deploy, LanceDB for
zero-key local runs, pgvector optional.

## Files

| File               | Role                                                              |
| ------------------ | ---------------------------------------------------------------- |
| `functions.py`     | `@connector.register_query def search_documents(...) -> list[SearchHit]` |
| `requirements.txt` | `ndc-sdk-python==0.42` + the `retrieval-bridge[turbopuffer]` package |
| `.env.example`     | Backend/embedder selection + turbopuffer credentials             |

`SearchHit` is a named pydantic object with explicit scalar fields (`id`, `score`,
`text`, and the ticket metadata). That named type with a concrete `id` is what
lets the command-result -> Accounts (Postgres) **relationship** attach in the
generated metadata. The docstring on `search_documents` is the planner-facing
Command description.

## How it maps to DDN

```bash
# 1. Scaffold a Python lambda connector named "search" in this subgraph.
#    Choose the hasura/python connector when prompted.
ddn connector init search -i

# 2. Drop these files in (functions.py, requirements.txt). The CLI generates the
#    connector's Dockerfile / connector.yaml around them.

# 3. Register env vars (do NOT commit secrets; .env.example documents the keys):
ddn connector env add search --env RETRIEVAL_BRIDGE_BACKEND=turbopuffer
ddn connector env add search --env RETRIEVAL_BRIDGE_EMBEDDER=local
ddn connector env add search --env TURBOPUFFER_API_KEY=tpuf_xxx
ddn connector env add search --env TURBOPUFFER_REGION=gcp-us-central1
ddn connector env add search --env TURBOPUFFER_NAMESPACE=retrieval-bridge-tickets

# 4. Introspect to generate the Command + SearchHit object type, then build:
ddn connector introspect search
ddn command add search "*"
ddn supergraph build local   # or: ddn supergraph build create
```

Once the `SearchHit` object type exists, define the relationship from its `id`
to the Accounts Postgres model in the metadata so PromptQL can join account/plan
facts onto each retrieved ticket.

## Local vs. production backend

The base `retrieval-bridge` install bundles fastembed (local BGE-small, 384-dim)
and LanceDB, so the container *can* run hybrid search with no cloud keys. But the
DDN connector runtime is **stateless** — a LanceDB file written into the container
does not persist across restarts/scale-out and there is nothing to seed it. So:

- **Local / dev:** run the demo from the repo root (`python scripts/seed.py &&
  python scripts/demo.py`) where LanceDB persists on disk, or run
  `python functions.py` against a local turbopuffer namespace.
- **Production:** set `RETRIEVAL_BRIDGE_BACKEND=turbopuffer` and point
  `TURBOPUFFER_NAMESPACE` at a **pre-seeded** namespace (seed it once with
  `RETRIEVAL_BRIDGE_BACKEND=turbopuffer python scripts/seed.py`). The connector
  then only reads; the same `search_documents` Command is unchanged.
