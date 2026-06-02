# Architecture

`retrieval-bridge` wires **turbopuffer-style hybrid retrieval** into **Hasura PromptQL** as a single, stable command, behind a one-method backend seam so the vector store is swappable (turbopuffer headline · LanceDB clone-and-run default · pgvector optional). This document describes how the pieces fit, where the seams are and why they are clean, the data model that crosses them, and the one framing that justifies the whole design: **retrieval is a deterministic tool inside the plan, not the answer.**

It is written to be read alongside the code. Every claim below points at the file that implements it.

---

## 1. The end-to-end flow

One natural-language question turns into a **deterministic query plan**, and that plan calls hybrid retrieval as *one step among several*. The local demo ([`scripts/demo.py`](../scripts/demo.py)) executes exactly this division of labor, offline and zero-key, against whatever backend you seeded.

The headline incident the demo runs (verbatim from [`scripts/demo.py`](../scripts/demo.py)):

> "We just had a production incident: right after the 14:00 deploy, our search API started returning 503 Service Unavailable and requests timed out under heavy load. EU customers were hit hardest. Which past resolved incidents look like this, what was the root cause of each, and how were they fixed?"

The plan PromptQL generates for that question, and the demo executes step-by-step:

1. **NL → plan.** The PromptQL planner reads the question and the tool specs (the `description` fields in [`ddn/metadata/search_documents.hml`](../ddn/metadata/search_documents.hml)) and emits a Python query plan. The plan decides *to retrieve first*, with a `status=resolved` facet, because the user asked for "past resolved incidents."

2. **`search_documents` command → hybrid retrieval.** The plan calls the command:
   `search_documents(query=<incident>, top_k=8, filters={status: "resolved"})`.
   This is the deploy-side lambda ([`ddn/connector/search/functions.py`](../ddn/connector/search/functions.py)) whose body is `RetrievalBridge.search_documents` ([`retrieval_bridge/search.py`](../retrieval_bridge/search.py)) — the *same code* the demo runs in-process. The bridge embeds the query once ([`embed_query`](../retrieval_bridge/embedders/local_bge.py)) and hands `(query_embedding, query_text, top_k, filters)` to the active backend.

3. **Swappable `VectorBackend` → two sub-queries.** The backend ([`VectorBackend` protocol](../retrieval_bridge/backends/base.py)) runs a **vector ANN** sub-query (meaning) and a **BM25 full-text** sub-query (exact tokens), each with the filter applied, each returning a ranked candidate list. turbopuffer does both in one snapshot-isolated `multi_query`; LanceDB and pgvector issue the two queries separately. The store does not fuse — by design.

4. **RRF hybrid fusion.** The two ranked lists are fused **client-side** with [`reciprocal_rank_fusion`](../retrieval_bridge/fusion.py), shared by every backend so "hybrid" means the same thing everywhere. The top `top_k` fused `Hit`s come back. *Measured here:* the 503 incident returns **7 exact-503 tickets** (won by BM25) **plus 1 semantically-related DB-pool-exhaustion ticket** (surfaced by the vector side) — the hybrid catching both the literal `503` and the conceptual "saturation under load."

5. **Artifact.** The plan stores the hits as a **PromptQL artifact** — structured data held *outside* the LLM's context window — so later steps operate on rows, not on re-pasted prose.

6. **`classify` / `summarize` per hit.** For each retrieved ticket the plan calls focused LLM primitives: `executor.classify(...)` to label the root cause and `executor.summarize(...)` to extract the fix. In the demo these are emulated deterministically by [`classify_root_cause` / `summarize_fix`](../retrieval_bridge/demo_primitives.py). *Measured here:* every classified root cause matched the ground-truth label carried in ticket metadata.

7. **DDN relationship join.** Each hit's `id` is joined to its structured account row — project, plan tier, monthly revenue — through a **declarative DDN relationship** ([`ddn/metadata/relationship_searchhit_account.hml`](../ddn/metadata/relationship_searchhit_account.hml)): `SearchHit.id → Accounts.ticket_id`, no join code. The demo emulates this with [`StructuredStore.get_by_ids`](../retrieval_bridge/structured.py) over SQLite.

8. **Deterministic rank.** Plain Python sorts the enriched rows by **plan tier, then recency** ([`scripts/demo.py`](../scripts/demo.py), `PLAN_RANK`). *Measured here:* the scale-tier accounts (Pavo Retail $12.4k MRR, Lyra Media $7.2k MRR) are promoted above the launch-tier tickets.

The final answer is *grounded* (every claim traces to a retrieved ticket), *enriched* (root cause + fix), *prioritized by business value* (the join), and *reproducible* (steps 5–8 are deterministic; only steps 6's primitives use an LLM, and only in tightly scoped contexts).

---

## 2. Architecture diagram

```
                        ┌──────────────────────────────────────────────┐
   User (natural        │                  PromptQL                     │
   language question) ─►│  ┌────────────┐      generates       ┌──────┐ │
                        │  │  PLANNER   │ ───────────────────► │ PLAN │ │
                        │  └────────────┘   (a Python program) └──┬───┘ │
                        │                                          │     │
                        │   RUNTIME (deterministic execution)      │     │
                        │   • artifacts (data kept OUT of context) │     │
                        │   • executor.classify / summarize  ◄─────┤     │
                        │                                          │     │
                        └──────────────────────────────────────┬──┼─────┘
                                                                │  │
                          calls command  search_documents(query, top_k, filters)
                                                                │  │
                        ┌───────────────────────────────────── ▼ ─┼─────┐
              SEAM 3 ──►│  Hasura DDN  ──  Python lambda connector  │     │
            (connector) │  ddn/connector/search/functions.py        │     │
                        │   returns  list[SearchHit]  (named type)  │     │
                        └───────────────────────────────┬──────────┼─────┘
                                                         │          │
                                  RetrievalBridge.search_documents()│
                                  retrieval_bridge/search.py        │
                                                         │          │
                       SEAM 1 ──► embed_query(text) ─────┤          │
                      (Embedder)  embedders/base.py      │          │
                                                         ▼          │
                       SEAM 2 ──►       VectorBackend (one interface)│
                    (VectorBackend)     backends/base.py            │
                          ┌──────────────────┼──────────────────┐  │
                          ▼                   ▼                  ▼  │
                    turbopuffer            LanceDB           pgvector│
                   (headline)         (open-source        (DDN-native│
                                       analog / local)      Postgres)│
                          │                   │                  │   │
                    vector ANN  +  BM25/FTS sub-queries (per backend)│
                          └──────────────────┬──────────────────┘   │
                                             ▼                       │
                          reciprocal_rank_fusion()  (CLIENT-SIDE,    │
                          retrieval_bridge/fusion.py — shared)       │
                                             │                       │
                                       list[Hit]  ──────────────────►┘
                                             │
                              (hit.id is the join key)
                                             ▼
                        ┌──────────────────────────────────────────┐
                        │  Postgres  Model: Accounts                │
                        │  (project · plan_tier · monthly_revenue)  │
                        │  joined to each hit by ticket_id via the  │
                        │  declarative DDN relationship             │
                        │  (local analog: StructuredStore / SQLite) │
                        └──────────────────────────────────────────┘
```

The three labeled seams (**Embedder**, **VectorBackend**, **DDN lambda connector**) are the load-bearing boundaries. Everything to the left of a seam is unaware of what is on the right; that is what makes the system swappable and what makes the local demo a faithful mirror of the deploy.

---

## 3. The three seams (and why each is clean)

A clean seam means: a small, explicit interface; both sides depend only on the interface; you can replace one side without touching the other; and the *same* implementation can serve both the local demo and the deployed system.

### Seam 1 — `Embedder` ([`retrieval_bridge/embedders/base.py`](../retrieval_bridge/embedders/base.py))

```python
class Embedder(Protocol):
    name: str          # stable model id -> per-embedder namespace
    dim: int           # must equal the backend's declared vector dimension
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
```

**Why two methods, not one.** Retrieval-tuned cloud embedders are *asymmetric*: Voyage takes `input_type="document"` vs `"query"`; Cohere takes `"search_document"` vs `"search_query"`. A single symmetric `embed()` would silently degrade recall on those providers. Splitting write-side from query-side lets each implementation apply the correct per-side hint while using the **same model** on both sides — which is non-negotiable for a vector store like turbopuffer, where a namespace's dimension and distance metric are *fixed at creation*, so write-time and query-time vectors must match exactly.

**Why it is a clean boundary.** The default is zero-key — fastembed BGE-small, 384-dim ([`local_bge.py`](../retrieval_bridge/embedders/local_bge.py)) — chosen because it is the model turbopuffer uses in its own vector-search guide, so the local default mirrors the documented turbopuffer path. Cloud embedders ([`openai`](../retrieval_bridge/embedders/openai_embedder.py), [`voyage`](../retrieval_bridge/embedders/voyage_embedder.py), [`cohere`](../retrieval_bridge/embedders/cohere_embedder.py)) are lazily imported extras selected by `RETRIEVAL_BRIDGE_EMBEDDER` ([`embedders/__init__.py`](../retrieval_bridge/embedders/__init__.py)). The base install needs no cloud SDKs. The bridge holds only an `Embedder`; it never names a model.

### Seam 2 — `VectorBackend` ([`retrieval_bridge/backends/base.py`](../retrieval_bridge/backends/base.py))

```python
class VectorBackend(Protocol):
    name: str
    def upsert(self, docs: list[dict[str, Any]]) -> None: ...
    def search(self, query_embedding, query_text, top_k=10, filters=None) -> list[Hit]: ...
```

**This is the heart of the pitch.** Every backend takes the same inputs and returns the same `list[Hit]`, so the PromptQL-facing `search_documents` command never changes when you swap the store. `search` receives *both* the query embedding (for the vector sub-query) and the raw query text (for the BM25 sub-query); each backend runs both and fuses them with the shared [`reciprocal_rank_fusion`](../retrieval_bridge/fusion.py).

**Why it is a clean boundary.** The surface is deliberately the one turbopuffer exposes — vector + BM25 + *client-side* RRF — so `TurbopufferBackend` is a faithful default rather than a special case, and LanceDB / pgvector are honest stand-ins, not toys. The key design fact: **turbopuffer has no server-side rank fusion** (verified against its hybrid docs). Hybrid search there is a `multi_query` of one vector sub-query and one BM25 sub-query against a single consistent snapshot, fused on the client. So RRF lives in the bridge for *all* backends — and swapping stores changes only *where the two sub-queries run*, never what "hybrid" means. The backend-agnostic `filters` dict (equality, membership, operator dicts) is translated by each backend into its own dialect: turbopuffer's tuple DSL ([`_build_filters`](../retrieval_bridge/backends/turbopuffer_backend.py)), LanceDB SQL ([`_build_where`](../retrieval_bridge/backends/lancedb_backend.py)), or a parameterized Postgres `WHERE` over JSONB ([`_build_where`](../retrieval_bridge/backends/pgvector_backend.py)). Selection is `RETRIEVAL_BRIDGE_BACKEND`, with optional SDKs imported lazily ([`backends/__init__.py`](../retrieval_bridge/backends/__init__.py)).

### Seam 3 — DDN lambda connector ([`ddn/connector/search/functions.py`](../ddn/connector/search/functions.py))

```python
@connector.register_query
def search_documents(query, top_k=5, plan_tier=None, status=None) -> list[SearchHit]:
    ...
```

This is the seam between PromptQL and retrieval. The connector is a thin wrapper that (a) translates the planner-supplied facets `plan_tier` / `status` into the bridge's backend-agnostic filter dict, (b) calls `RetrievalBridge.search_documents`, and (c) flattens each `Hit` into a **named** `SearchHit` object. The bridge it wraps is built lazily and cached module-level (`_get_bridge`) so the heavy embedder model and backend client are imported once and reused across calls.

**Why it is a clean boundary — three DDN-specific reasons:**

1. **DDN never learns which store answered.** The connector reads no backend config of its own; it constructs `RetrievalBridge()`, which resolves backend/embedder from env. turbopuffer for the headline deploy, LanceDB for zero-key local, pgvector optional — the Command signature in [`search_documents.hml`](../ddn/metadata/search_documents.hml) (`outputType: "[SearchHit!]!"`) is unchanged either way.
2. **The return type must be a *named* object with a concrete `id`.** DDN infers the NDC object type from the Python type hints. Only a *named* type (not `dict`/`Any`) can be the target of a relationship, and only a concrete `id` field gives the `SearchHit → Accounts` join something to map from. This is why `SearchHit` is a flat pydantic model with explicit scalar fields, and why `Hit` ([`types.py`](../retrieval_bridge/types.py)) is a small explicit model rather than a bare dict.
3. **The docstring is a tool spec for the planner.** The function docstring becomes the initial Command `description`, which is then hand-tuned in [`search_documents.hml`](../ddn/metadata/search_documents.hml). PromptQL's tool selection is driven by that text — a vague description means the planner won't reach for hybrid retrieval when the user says "find tickets like this one." The description deliberately spells out *what it does, when to use it, what the args mean, and that an `account` relationship is available to join.*

The deploy maps to DDN with no bespoke plumbing: `ddn connector init search`, drop in `functions.py`, register env, `ddn connector introspect` (which *generates* `SearchHit.hml` and a draft `search_documents.hml`), then add the relationship by hand. See [`ddn/connector/search/README.md`](../ddn/connector/search/README.md).

---

## 4. Data model

Three object shapes cross the seams. They are intentionally small and named.

### `Hit` — the universal retrieval result ([`retrieval_bridge/types.py`](../retrieval_bridge/types.py))

The single object that crosses every seam: `backend.search(...) -> list[Hit] -> RetrievalBridge.search_documents -> the DDN command -> artifact rows`.

| Field | Type | Role |
|---|---|---|
| `id` | `str` | Stable document id; **the join key** to structured data. |
| `score` | `float` | The *fused* RRF relevance (higher is better). Ranking-only — not comparable across backends in absolute terms. |
| `text` | `str` | The retrieved document text (the ticket body). |
| `metadata` | `dict[str, Any]` | Filterable attributes: `project_id`, `plan_tier`, `status`, `created_at`, `component`, `error_code`, `severity`, `root_cause`. |

It is a pydantic model, not a dict, because the connector's `SearchHit` is derived from this shape and DDN needs a named, `id`-bearing type for the relationship to attach.

### Tickets — the indexed corpus ([`data/generate_corpus.py`](../data/generate_corpus.py))

160 synthetic support tickets (deterministic, seeded), engineered to exercise *hybrid* retrieval specifically — every ticket carries **both** semantic content (rewards vector search) **and** exact tokens (rewards BM25):

- **Semantic** — naturally-worded symptoms that *imply* a root cause ("service is down under load", "first query each morning is slow", "users signed out unexpectedly"). The canonical `root_cause` label is kept in metadata for evaluation, never pasted into the body.
- **Exact tokens** — distinctive literals BM25 pins: error codes (`ERR_DIM_384`, `OOMKilled`, `ERR_TLS_526`), ticket IDs (`TCK-10042`), HTTP statuses (`503`, `429`), plan-tier names.

The embedded/indexed `text` folds subject + body + (when resolved) the `Resolution:` narrative into one searchable document. Ten incident themes plus a "question, not an incident" noise class. `~70%` of tickets are resolved/closed so the demo has fixed incidents to learn from. The seed script ([`scripts/seed.py`](../scripts/seed.py)) carries this flat attribute set into the vector store (`ATTRS`): `project_id, plan_tier, status, created_at, component, error_code, severity, root_cause`.

### Accounts — the structured facts ([`structured.py`](../retrieval_bridge/structured.py) / [`accounts.hml`](../ddn/metadata/accounts.hml))

The "facts" each ticket joins to, keyed by `ticket_id`. In the deploy this is a Postgres table exposed as a DDN Model; locally it is the SQLite `accounts` table — column-for-column identical:

| Column | Type | Role |
|---|---|---|
| `ticket_id` | `TEXT` PK | **Join target** for `SearchHit.id`. |
| `project_id` | `TEXT` | Stable project identifier. |
| `project_name` | `TEXT` | Human-readable customer name. |
| `plan_tier` | `TEXT` | `free \| launch \| scale \| enterprise` — used to rank hits. |
| `monthly_revenue` | `REAL` | MRR (USD) — used to prioritize high-value customers. |
| `account_region` | `TEXT` | Hosting region. |
| `seat_count` | `INTEGER` | Seats on the account. |

`plan_tier` is the authoritative account tier; it is *also* denormalized onto the ticket as a filterable attribute, so retrieval can pre-filter by tier without a join, and the join still supplies revenue/name for ranking.

---

## 5. Local demo ↔ deployed system (1:1 mapping)

The repo's core ([`retrieval_bridge/`](../retrieval_bridge)) is shared verbatim by the demo and the connector. The only things the demo *emulates* are the two pieces that genuinely require external systems (a PromptQL runtime and a Postgres). Each emulation is labeled at its call site and maps 1:1 to its production counterpart.

| Demo component (local, zero-key) | Production component (deployed) | Same code? |
|---|---|---|
| `RetrievalBridge.search_documents` called in-process ([`search.py`](../retrieval_bridge/search.py)) | The body of the DDN lambda Command ([`functions.py`](../ddn/connector/search/functions.py)) | **Identical** |
| `VectorBackend = LanceDBBackend` (env default) | `VectorBackend = TurbopufferBackend` (`RETRIEVAL_BRIDGE_BACKEND=turbopuffer`) | Same interface, swapped impl |
| `Embedder = LocalBGEEmbedder` (fastembed, 384-dim) | Same local BGE, or a cloud embedder via env | Same interface |
| `reciprocal_rank_fusion` client-side ([`fusion.py`](../retrieval_bridge/fusion.py)) | Same function, same place (turbopuffer has no server-side fusion) | **Identical** |
| `Hit` ([`types.py`](../retrieval_bridge/types.py)) | `SearchHit` named NDC object type ([`SearchHit.hml`](../ddn/metadata/SearchHit.hml)) | Hit's fields hoisted to top level |
| `classify_root_cause` ([`demo_primitives.py`](../retrieval_bridge/demo_primitives.py)) | `executor.classify(...)` — focused LLM primitive in the PromptQL runtime | Emulated (labeled) |
| `summarize_fix` ([`demo_primitives.py`](../retrieval_bridge/demo_primitives.py)) | `executor.summarize(...)` — focused LLM primitive | Emulated (labeled) |
| `StructuredStore.get_by_ids` over SQLite ([`structured.py`](../retrieval_bridge/structured.py)) | `SearchHit.account` declarative DDN relationship → Postgres `Accounts` Model ([`relationship_searchhit_account.hml`](../ddn/metadata/relationship_searchhit_account.hml)) | Emulated (same lookup) |
| Python `sort(key=plan_tier, recency)` ([`demo.py`](../scripts/demo.py)) | Deterministic ranking step inside the PromptQL plan | Same logic |

One subtlety worth calling out for the deploy: the DDN connector runtime is **stateless**, so an embedded LanceDB file written into a container does not persist across restarts/scale-out. Local/dev uses LanceDB on disk (or a local turbopuffer namespace); production sets `RETRIEVAL_BRIDGE_BACKEND=turbopuffer` and points `TURBOPUFFER_NAMESPACE` at a **pre-seeded** namespace, after which the connector is read-only. Details in [`ddn/connector/search/README.md`](../ddn/connector/search/README.md).

---

## 6. Swappable-backend matrix

All three implement the same [`VectorBackend`](../retrieval_bridge/backends/base.py) protocol and reuse the same [`reciprocal_rank_fusion`](../retrieval_bridge/fusion.py). "Hybrid" therefore means the same thing on each — two ranked sub-queries fused client-side.

| Backend | Install | Vector sub-query | Full-text sub-query | Filters | Fusion | Role |
|---|---|---|---|---|---|---|
| **turbopuffer** ([impl](../retrieval_bridge/backends/turbopuffer_backend.py)) | `.[turbopuffer]` | `rank_by=("vector","ANN",…)` in a snapshot-isolated `multi_query` | `rank_by=("text","BM25",…)` in the same `multi_query` | tuple DSL (`Eq`/`In`/`Gte`…) | client-side RRF | **Headline default** — serverless, object-storage-native hybrid at scale |
| **LanceDB** ([impl](../retrieval_bridge/backends/lancedb_backend.py)) | base (default) | `tbl.search(vec)` ANN | `tbl.search(text, query_type="fts")` native BM25 | LanceDB SQL `WHERE` (prefilter) | client-side RRF | **Zero-infra clone-and-run** — the open-source turbopuffer analog |
| **pgvector** ([impl](../retrieval_bridge/backends/pgvector_backend.py)) | `.[pgvector]` | cosine `embedding <=> %s` (IVFFlat) | `to_tsvector @@ plainto_tsquery`, ranked by `ts_rank` (GIN) | parameterized SQL over JSONB `metadata->>` | client-side RRF | **DDN-native locality** — corpus + Accounts can share one Postgres, so the hit→account join can be a native SQL join |

Pattern shared by all three `search` methods: fetch `candidates = max(top_k * 4, 20)` per sub-query, fuse, return the top `top_k`. Each carries a per-modality raw score for debugging only; RRF re-scores purely by rank, so absolute backend scores never leak into ranking.

**Why hybrid, concretely** — the benchmark ([`scripts/bench.py`](../scripts/bench.py), measured):

- **Section A — exact-identifier lookup (vector's blind spot).** Looking up one specific ticket id among 160 look-alikes. Embeddings map all such tokens to nearly the same place, so pure-vector flounders: **MRR 0.117, found@5 = 1/12**. BM25 pins the literal; **hybrid recovers it: MRR 0.572, found@5 = 12/12.**
- **Section B — semantic queries (vector's strength).** Paraphrased problems with no exact tokens. Vector and hybrid **both hit 5/5 @5** — hybrid does not regress. You get the keyword win for free.

That is exactly why turbopuffer pairs BM25 with vectors, and why the bridge fuses them regardless of store.

---

## 7. Retrieval is a tool inside the plan — not the whole answer

This is the framing PromptQL stands for, and it is the reason this artifact wires retrieval in as a *command* rather than as the answer engine.

**Naive RAG** is a fixed two-step pipeline: embed the question → top-k vector search → stuff the chunks into the prompt → let the LLM free-associate over them. The retrieval step *is* the architecture. Its failure modes are structural:

- **Retrieval is the whole answer.** If recall misses, the LLM hallucinates over whatever came back; if it over-returns, the context fills with noise. There is no step that *reasons about* the candidate set.
- **No exact-token recall.** Pure-vector RAG is blind to error codes, IDs, and SKUs — the very tokens support and ops questions hinge on (Section A above: MRR 0.117).
- **No deterministic structure.** Ranking, joining business facts, classifying, prioritizing — all of it gets smeared into one ungrounded generation. Re-running the same question can give a different answer.
- **No structured join.** A retrieved chunk cannot be reliably joined to authoritative facts (this account's plan, this customer's MRR); the model is asked to "remember" or infer them.

**The plan model** inverts this. The PromptQL plan is a *Python program*, not one prompt:

- **Retrieval is one deterministic step.** `search_documents` narrows millions of documents to a handful of strong candidates — and then hands off. The retrieval comment in [`search.py`](../retrieval_bridge/search.py) says it plainly: *"intentionally the whole retrieval step … after which PromptQL does the planning, joining, and reasoning."*
- **It is *hybrid*, so exact tokens are first-class.** Vector for meaning + BM25 for literals, fused — the structural fix for RAG's exact-token blind spot.
- **Artifacts keep data out of the context window.** Hits are stored as structured rows and operated on as data, not re-pasted as prose — so the LLM reasons over a table, not a wall of text.
- **LLM judgment is scoped to where it is needed.** `classify` and `summarize` run in isolated, focused contexts over *one ticket at a time*, with deterministic everything-else around them.
- **Structured facts join declaratively.** `hit.id → Accounts` resolves authoritative plan/revenue with no join code and no model guesswork ([relationship](../ddn/metadata/relationship_searchhit_account.hml)).
- **Ranking is explicit and reproducible.** "By plan tier, then recency" is a sort, not a vibe — so the scale-tier accounts land above launch every run.

The demo's closing "division of labor" makes the contrast literal:

> **turbopuffer / LanceDB** → scalable, cheap **hybrid retrieval** (millions → a handful)
> **PromptQL** → **planning + deterministic execution** + `classify`/`summarize`
> **Postgres via DDN** → **structured facts** joined to hits by a declarative relationship

Each layer does the one thing it is best at. Retrieval's job is to be a high-recall, low-cost *funnel* feeding a deterministic plan — and because that funnel sits behind the `VectorBackend` seam and the `search_documents` command, you can make it turbopuffer, LanceDB, or pgvector without the plan above it noticing.

---

## See also

- [`README.md`](../README.md) — the pitch and quickstart.
- [`retrieval_bridge/search.py`](../retrieval_bridge/search.py) — the command body (shared by demo + connector).
- [`retrieval_bridge/fusion.py`](../retrieval_bridge/fusion.py) — the client-side hybrid step.
- [`ddn/connector/search/functions.py`](../ddn/connector/search/functions.py) — the deploy-side lambda.
- [`ddn/metadata/`](../ddn/metadata) — the Command, the `SearchHit` type, the `Accounts` model, and the relationship that joins them.
