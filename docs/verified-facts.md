# Verified facts — every external API checked against live docs (June 2026)

This is the receipts file. Every external API this repo builds on was verified
against **live vendor documentation in June 2026**, not against model training
memory — because the APIs in this stack have all drifted, and several drifted in
ways that would have silently broken the build if taken from memory.

This document is organized by area. Each area lists:

- **Build-critical facts** — what the code actually depends on, with a source URL.
- **Brief-vs-reality corrections** — where the original job-pitch brief (or a
  plausible-from-memory assumption) was *wrong*, and what the live docs say
  instead. These are the ones that matter: they are the drift that memory would
  have gotten wrong.

> **Why this matters as a signal.** Anyone can paste an SDK snippet from memory.
> The four corrections below (turbopuffer v2 + region hosts, `ndc_sdk_python`
> imports, no server-side fusion, Voyage `voyage-3` deprecation) are exactly the
> kind of breaking drift that ships a broken connector. Verifying against live
> docs caught them before they cost anything.

A model-string caveat that applies everywhere: **dated model snapshots
(`claude-sonnet-4-5-20250929`, `voyage-3.5`, `embed-v4.0`, …) are point-in-time
and will be superseded.** Treat the *shapes* (dimensions, symmetric vs
asymmetric, the API surface) as the durable facts; re-check the exact id string
before any deploy.

---

## 1. turbopuffer (the headline retrieval backend)

### Build-critical facts

| Fact | Detail | Source |
|---|---|---|
| API version | **v2** is the current API. The legacy `api.turbopuffer.com` global host is dead — do not use it. | <https://turbopuffer.com/docs/overview> |
| Host is region-scoped | Every request goes to `https://{region}.turbopuffer.com`. There is **no global endpoint**, and the **API key is tied to the region it was created in**. | <https://turbopuffer.com/docs/regions> |
| Client constructor | `Turbopuffer(region=..., api_key=...)` — region is a first-class constructor arg, not a base-URL override. | <https://pypi.org/project/turbopuffer/> |
| Write call | Documents are written with **`upsert_rows`** (row-oriented upsert), not a generic `upsert`/`insert`. Namespace is **created implicitly on first write**. | <https://turbopuffer.com/docs/write> |
| Hybrid query call | `multi_query` runs multiple sub-queries (vector ANN + BM25) in **one snapshot-isolated request**. | <https://turbopuffer.com/docs/query> |
| `multi_query` cap | At most **16 sub-queries** per `multi_query` call. (The hybrid demo uses 2: one vector, one BM25 — well inside the cap.) | <https://turbopuffer.com/docs/query> |
| BM25 needs a schema flag | Full-text/BM25 ranking requires the text column be declared with a **`full_text_search`** schema; it is not on by default. The vector column is `[N]f32` with `ann: true`. | <https://turbopuffer.com/docs/hybrid> |
| Latency profile | **Cold** read (namespace not recently touched, served from object storage): **low hundreds of ms**. **Warm**: **~14 ms**. | <https://turbopuffer.com/docs/architecture> |
| Pricing floor | **No free tier.** Lowest plan is **Launch = $64/month minimum** (pay the greater of metered usage or $64); next tier **Scale = $256/mo**. 30-day cancel-for-refund window. | <https://turbopuffer.com/docs/limits> |

### Brief-vs-reality corrections

- **No server-side rank fusion → RRF is client-side.** turbopuffer returns the
  two ranked result sets from a `multi_query`; it does **not** fuse them
  server-side. The brief's "hybrid" must therefore be completed in the client:
  the bridge does reciprocal-rank fusion in `retrieval_bridge/fusion.py`. This is
  load-bearing for the anti-lock-in claim — every backend (LanceDB, pgvector)
  mirrors the *same* client-side RRF, so "hybrid" means the identical thing
  everywhere.
  Source: <https://turbopuffer.com/docs/hybrid>
- **Region is part of the host, not a header.** A from-memory assumption of a
  single `api.turbopuffer.com` with a region header is wrong — the region is in
  the hostname (`{region}.turbopuffer.com`) and binds the key. A region/key
  mismatch fails to authenticate.
  Source: <https://turbopuffer.com/docs/regions>
- **There is no free tier to demo on.** This is why the repo defaults to local
  LanceDB and treats turbopuffer as a one-env-var upgrade — see
  [turbopuffer-runbook.md](turbopuffer-runbook.md).
  Source: <https://turbopuffer.com/docs/limits>

---

## 2. Hasura DDN — the Python lambda connector

### Build-critical facts

| Fact | Detail | Source |
|---|---|---|
| SDK package | **`ndc-sdk-python==0.42`** (pinned in `ddn/connector/search/requirements.txt`). | <https://github.com/hasura/ndc-python-lambda> |
| Import root | Imports come from **`ndc_sdk_python`**, e.g. `from ndc_sdk_python import start` and `from ndc_sdk_python.function_connector import FunctionConnector`. | <https://github.com/hasura/ndc-python-lambda> |
| Connector object | `connector = FunctionConnector()`; functions are exposed with the **`@connector.register_query`** decorator (use `register_mutation` for writes). | <https://hasura.io/docs/3.0/business-logic/add-business-logic/> |
| Return type | Functions return a **named pydantic `BaseModel`** with explicit scalar fields (not `dict`/`Any`). DDN infers the NDC object type from the type hints; only a named type with a concrete `id` field lets the result→Accounts relationship attach. | <https://hasura.io/docs/3.0/business-logic/return-functions/> |
| Description source | The function **docstring becomes the planner-facing Command description**, so it is written for the query planner, not for code readers. | <https://hasura.io/docs/3.0/business-logic/> |
| Runtime | The connector runs on **Python 3.12**. | <https://github.com/hasura/ndc-python-lambda> |
| CLI | The **`ddn` CLI is v4**; `ddn connector init search -i` scaffolds the connector. | <https://hasura.io/docs/3.0/cli/> |

### Brief-vs-reality corrections

- **Imports are `ndc_sdk_python`, NOT `hasura_ndc`.** A from-memory `import
  hasura_ndc` (the older naming) does not exist for the current lambda
  connector — the package is `ndc-sdk-python` and the import root is
  `ndc_sdk_python`. This is the single most likely thing memory gets wrong here.
  Source: <https://github.com/hasura/ndc-python-lambda>
- **The return type must be a *named* pydantic model, not a bare dict.** A
  function returning `dict`/`Any` produces no named NDC object type, and without
  a named type with an `id` field the `SearchHit.account` relationship has
  nothing to attach to. The connector therefore returns `SearchHit(BaseModel)`.
  Source: <https://hasura.io/docs/3.0/business-logic/return-functions/>

---

## 3. DDN relationships — joining a command result to a model

### Build-critical facts

| Fact | Detail | Source |
|---|---|---|
| Command-result→model joins are supported | A relationship can be declared **from a command's return type to a Model**, declaratively, with no join code. | <https://hasura.io/docs/3.0/data-modeling/relationship/> |
| `sourceType` is the return-type name | `sourceType: SearchHit` — the **ObjectType the command returns**, *never* the command name `search_documents`. Relationships attach to a type. | <https://hasura.io/docs/3.0/reference/metadata-reference/relationships/> |
| `relationshipType: Object` | Use **`Object`** when the target Model resolves to at most one row per source (here `ticket_id` is unique → one account per hit). Use `Array` only for one-to-many. | <https://hasura.io/docs/3.0/reference/metadata-reference/relationships/> |
| Mapping | `source.fieldPath: id` → `target.modelField: ticket_id`. | <https://hasura.io/docs/3.0/data-modeling/relationship/> |

### Brief-vs-reality corrections

- **This relationship is hand-authored — `ddn relationship add` will NOT
  generate it.** The auto-generator only produces relationships from **Postgres
  foreign keys between two Postgres Models**. Here the source is a **command
  return type** (`SearchHit`, from the Python connector), there is no FK, so the
  `.hml` is written by hand (`ddn/metadata/relationship_searchhit_account.hml`).
  Source: <https://hasura.io/docs/3.0/data-modeling/relationship/>
- **`sourceType` is the type, not the command.** Pointing `sourceType` at
  `search_documents` (the command) instead of `SearchHit` (its return type) is a
  natural mistake that does not validate — relationships live on types.
  Source: <https://hasura.io/docs/3.0/reference/metadata-reference/relationships/>

---

## 4. PromptQL

### Build-critical facts

| Fact | Detail | Source |
|---|---|---|
| AI primitives | There are **four**: **`summarize`, `classify`, `extract`, `visualize`** — focused LLM calls the plan invokes only where judgment is needed. | <https://promptql.io/docs/ai-primitives/> |
| Config kind/version | **`kind: PromptQlConfig`, `version: v2`** — with an `llm` (planner) and an `aiPrimitivesLlm` (primitives), each `provider: anthropic` + dated `model` + `apiKey.valueFromEnv`. | <https://promptql.io/docs/metadata/> |
| Tool selection | The planner selects tools via their **metadata `description`** fields (command docstrings + relationship descriptions), steered by `systemInstructions`. | <https://promptql.io/docs/metadata/> |
| How to run it | Via the **Playground**, the **natural-language query API**, or the **SDK**. | <https://promptql.io/docs/promptql-apis/> |

### Brief-vs-reality corrections

- **There is no `promptql` CLI to run a plan.** Plans are run from the
  **Playground**, the **NL query API**, or the **SDK** — not a standalone
  `promptql run` command. (The `ddn` CLI manages the project/metadata; it is not
  the thing that executes a query plan.)
  Source: <https://promptql.io/docs/promptql-apis/>
- **It is four primitives, not "RAG with extra steps."** The primitive set is
  exactly `summarize / classify / extract / visualize`; retrieval
  (`search_documents`) is a *tool inside the plan*, not a fifth primitive.
  Source: <https://promptql.io/docs/ai-primitives/> and
  <https://promptql.io/docs/reference/>

---

## 5. Embeddings (the pluggable embedder layer)

The repo default is **local fastembed BGE-small (384-dim)** — chosen because it
is exactly the model in turbopuffer's own getting-started guide, so the local
demo matches the production recipe with zero keys. Cloud embedders are pluggable.

### Build-critical facts

| Provider | Model | Dim | Symmetric? | Notes | Source |
|---|---|---|---|---|---|
| **fastembed (local, default)** | BGE-small-en-v1.5 | **384** | symmetric | turbopuffer's own recommended starter model; zero API keys. | <https://turbopuffer.com/docs/overview> |
| **OpenAI** | `text-embedding-3-small` | **1536** | **symmetric** (same model for query + doc) | No separate query/doc model. | <https://platform.openai.com/docs/guides/embeddings> |
| **Voyage** | `voyage-3.5` / `voyage-4` | **1024** | **asymmetric** (`input_type=query` vs `document`) | Pass `input_type` per side. | <https://docs.voyageai.com/docs/embeddings> |
| **Cohere** | `embed-v4.0` | **1024** | **asymmetric** (`input_type=search_query` vs `search_document`) | Pass `input_type` per side. | <https://docs.cohere.com/docs/embeddings> |

### Brief-vs-reality corrections

- **Voyage `voyage-3` is deprecated.** Use **`voyage-3.5`** (or `voyage-4`); a
  from-memory `voyage-3` points at a deprecated model. The current asymmetric
  models are 1024-dim.
  Source: <https://docs.voyageai.com/docs/embeddings>
- **Voyage and Cohere are asymmetric; OpenAI is symmetric.** This is not a
  cosmetic detail — asymmetric models require passing an `input_type` that
  differs between the query embedding and the document embedding, and getting it
  wrong quietly degrades recall. OpenAI uses one model for both sides.
  Sources: <https://docs.voyageai.com/docs/embeddings>,
  <https://docs.cohere.com/docs/embeddings>,
  <https://platform.openai.com/docs/guides/embeddings>
- **Cohere's current model is `embed-v4.0` (1024-dim).** Earlier `embed-english-v3.0`
  assumptions from memory are superseded.
  Source: <https://docs.cohere.com/docs/embeddings>

---

## Source index (all live-checked June 2026)

**turbopuffer** — overview, regions, write, query, hybrid, limits, architecture:
<https://turbopuffer.com/docs/overview> ·
<https://turbopuffer.com/docs/regions> ·
<https://turbopuffer.com/docs/write> ·
<https://turbopuffer.com/docs/query> ·
<https://turbopuffer.com/docs/hybrid> ·
<https://turbopuffer.com/docs/limits> ·
<https://turbopuffer.com/docs/architecture> ·
PyPI: <https://pypi.org/project/turbopuffer/>

**Hasura DDN business logic + connector**:
<https://hasura.io/docs/3.0/business-logic/> ·
<https://hasura.io/docs/3.0/business-logic/add-business-logic/> ·
<https://hasura.io/docs/3.0/business-logic/return-functions/> ·
<https://github.com/hasura/ndc-python-lambda> ·
<https://hasura.io/docs/3.0/cli/>

**DDN relationships**:
<https://hasura.io/docs/3.0/data-modeling/relationship/> ·
<https://hasura.io/docs/3.0/reference/metadata-reference/relationships/>

**PromptQL**:
<https://promptql.io/docs/metadata/> ·
<https://promptql.io/docs/ai-primitives/> ·
<https://promptql.io/docs/promptql-apis/> ·
<https://promptql.io/docs/reference/>

**Embeddings**:
<https://docs.voyageai.com/docs/embeddings> ·
<https://docs.cohere.com/docs/embeddings> ·
<https://platform.openai.com/docs/guides/embeddings>

---

### Related docs

- **[pitch.md](pitch.md)** — the full argument; cites these facts as "verified
  against live docs."
- **[turbopuffer-runbook.md](turbopuffer-runbook.md)** — the upgrade path that
  relies on the v2 API, region-scoped hosts, and pricing above.
- **[../README.md](../README.md)** — the two-minute, zero-key quickstart.
