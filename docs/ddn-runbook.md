# DDN runbook — wire `search_documents` into Hasura DDN + PromptQL

This is the copy-pasteable path from **"I only have browser PromptQL"** to a running
local PromptQL project where the planner can call our `search_documents` hybrid
retrieval command, then follow a declarative relationship to join Postgres account
facts onto every retrieved ticket.

It assumes you are starting from **zero scaffolding**: no DDN CLI installed, no
local project, just the `viper9503/retrieval-bridge` repo on disk and (optionally) a
browser PromptQL project. Every artifact you paste in already lives in this repo
under `ddn/` — this runbook tells you which file goes where and in what order.

> **The one idea to keep in mind.** DDN never learns *which* vector store answered.
> The connector calls `RetrievalBridge`, which reads `RETRIEVAL_BRIDGE_BACKEND`
> (turbopuffer for the headline deploy, LanceDB for zero-key local, pgvector
> optional) and runs the identical vector-ANN + BM25 sub-queries fused client-side
> with reciprocal-rank fusion. The `search_documents` Command signature never moves.

**There is no `promptql` CLI binary** — the CLI is **`ddn`**. PromptQL is a *config*
(`PromptQlConfig`) plus a *playground/API*, not a separate tool.

What you will end up with, in order:

0. [Prerequisites: Docker + the DDN CLI](#0-prerequisites)
1. [`ddn auth login`](#1-authenticate)
2. [`ddn supergraph init --with-promptql`](#2-initialize-the-supergraph)
3. [`ddn connector init search` (hasura/python) + drop in our files](#3-scaffold-the-python-connector-and-replace-its-files)
4. [`ddn connector env add` — turbopuffer secrets + backend selector](#4-register-connector-env-vars-secrets)
5. [`ddn connector introspect` + `ddn command add` + paste the rich description](#5-introspect-add-the-command-and-paste-the-rich-description)
6. [Add Postgres: connector, `accounts.sql`, `ddn model add`](#6-add-postgres-the-accounts-model)
7. [**Hand-add** the `SearchHit.account` relationship](#7-hand-add-the-searchhit--accounts-relationship)
8. [Configure `promptql-config.hml`](#8-configure-promptql)
9. [`ddn supergraph build local` + `ddn run docker-start` + `ddn console --local`](#9-build-run-and-open-the-playground)
10. [Ask the demo prompt](#10-ask-the-demo-prompt)
- [Appendix: connect to an existing cloud PromptQL project](#appendix-connect-to-an-existing-cloud-promptql-project)

---

## 0. Prerequisites

### Docker Compose v2.20+

The DDN local stack runs in Docker, and `ddn run docker-start` uses Compose. You need
**Docker Compose v2.20 or newer** (Docker Desktop ships it; on Linux install the
`docker-compose-plugin`).

```bash
docker --version
docker compose version    # must be v2.20.0 or newer
```

If `docker compose version` is older than 2.20, upgrade Docker Desktop / the
Compose plugin before going further — the build/run steps will fail otherwise.

### Install the DDN CLI (v4)

```bash
curl -L https://graphql-engine-cdn.hasura.io/ddn/cli/v4/get.sh | bash
```

Then verify your machine has everything the CLI needs (Docker, Compose version,
connectivity):

```bash
ddn doctor
```

Fix anything `ddn doctor` flags before continuing.

---

## 1. Authenticate

```bash
ddn auth login
```

This opens a browser to log in to your Hasura account. Use the **same account** that
owns your browser PromptQL project — that is how the CLI and the cloud project line
up later (see the [appendix](#appendix-connect-to-an-existing-cloud-promptql-project)).

---

## 2. Initialize the supergraph

Create a fresh project **with PromptQL enabled**. The `--with-promptql` flag wires in
the `PromptQlConfig` global we configure in step 8.

```bash
ddn supergraph init retrieval-bridge --with-promptql
cd retrieval-bridge
```

Notes:

- `--with-promptql` is the alpha PromptQL flag. If your org provisions data planes,
  you may also need `--project-data-plane-id <id>`; `ddn supergraph init` will tell
  you if it's required. A plain `ddn supergraph init retrieval-bridge` also works —
  you can add the `PromptQlConfig` by hand in step 8 either way.
- This scaffolds a standard DDN layout with a default subgraph named **`app`** and a
  `globals` subgraph. **All HML in this repo assumes `subgraph: app`** (see the
  relationship target in step 7). If you name your subgraph differently, change
  `subgraph: app` accordingly everywhere.
- From here on, **run every `ddn` command from inside the project directory** you just
  `cd`'d into. The retrieval-bridge repo (with the `ddn/` reference files) is a
  *separate* checkout — paths like `ddn/connector/search/functions.py` below refer to
  **that repo**, and you copy *from* it *into* this DDN project.

---

## 3. Scaffold the Python connector and replace its files

Scaffold a `hasura/python` lambda connector named **`search`**:

```bash
ddn connector init search -i
```

In the interactive prompts:

- **Connector type:** choose **`hasura/python`**.
- **Port:** accept the suggested port (or pick a free one).

This generates `app/connector/search/` containing `functions.py`, `requirements.txt`,
`.env`, `connector.yaml`, and a Dockerfile. **The CLI owns `connector.yaml`,
`.env`, and the Dockerfile** — leave those alone. You only replace the two code files.

### Replace `functions.py`

Overwrite the generated `app/connector/search/functions.py` with **our**
`ddn/connector/search/functions.py` from the retrieval-bridge repo:

```bash
# RB_REPO = absolute path to your retrieval-bridge checkout
cp "$RB_REPO/ddn/connector/search/functions.py" app/connector/search/functions.py
```

Why this file matters (it is the whole seam):

- It defines `search_documents(query, top_k, plan_tier, status) -> list[SearchHit]`,
  decorated with `@connector.register_query`. The body just calls
  `RetrievalBridge.search_documents` — the *same* code the local demo runs.
- `SearchHit` is a **named pydantic object with flat scalar fields** (not a `dict` /
  `Any`). DDN infers a named NDC ObjectType from those type hints, and **only a named
  type with a concrete `id` field** lets the `SearchHit → Accounts` relationship
  attach in step 7.
- The function **docstring becomes the Command's `description`** on introspect — but
  you will replace that draft with the richer, hand-tuned description in step 5.

### Merge `requirements.txt`

Our connector needs the retrieval-bridge package (with the turbopuffer extra) on top
of the SDK. Merge **our** `ddn/connector/search/requirements.txt` into the generated
one. The two required lines are:

```text
ndc-sdk-python==0.42
retrieval-bridge[turbopuffer] @ git+https://github.com/viper9503/retrieval-bridge.git
```

The simplest correct move is to overwrite the generated file with ours (it already
contains both lines):

```bash
cp "$RB_REPO/ddn/connector/search/requirements.txt" app/connector/search/requirements.txt
```

The base `retrieval-bridge` package pulls in **fastembed** (local BGE-small,
384-dim) and **LanceDB**, so the container *can* run hybrid search with no cloud
keys. The `[turbopuffer]` extra adds the turbopuffer v2 SDK so a production deploy can
point at a seeded namespace. The backend is chosen at runtime by env var (next step).

> **Stateless-container caveat.** The DDN connector runtime is stateless — a LanceDB
> file written *inside* the container does not survive restarts/scale-out and there is
> nothing to seed it. So for a real deploy, set the backend to **turbopuffer** and
> point it at a **pre-seeded namespace**. Seed it once from the repo with
> `RETRIEVAL_BRIDGE_BACKEND=turbopuffer python scripts/seed.py` (see
> [turbopuffer-runbook.md](turbopuffer-runbook.md)). The connector then only reads.

---

## 4. Register connector env vars (secrets)

`ddn connector env add` updates the connector's `.env`, `connector.yaml`, and the
Compose file together — **do not hand-edit those** and **do not commit secrets**. The
keys our bridge reads are documented in `ddn/connector/search/.env.example`.

Set the backend selector and the turbopuffer credentials:

```bash
ddn connector env add search --env RETRIEVAL_BRIDGE_BACKEND=turbopuffer
ddn connector env add search --env RETRIEVAL_BRIDGE_EMBEDDER=local
ddn connector env add search --env TURBOPUFFER_API_KEY=tpuf_xxxxxxxxxxxxxxxx
ddn connector env add search --env TURBOPUFFER_REGION=gcp-us-central1
ddn connector env add search --env TURBOPUFFER_NAMESPACE=retrieval-bridge-tickets
```

| Env var | Required? | Notes |
|---|---|---|
| `RETRIEVAL_BRIDGE_BACKEND` | **Yes** | `turbopuffer` for deploy. `lancedb` only works where a seeded file persists — not in the stateless container. `pgvector` optional. |
| `RETRIEVAL_BRIDGE_EMBEDDER` | No (default `local`) | `local` = fastembed BGE-small (384-dim), no key. `openai`/`voyage`/`cohere` are pluggable (need their own keys). |
| `TURBOPUFFER_API_KEY` | **Yes** (turbopuffer backend) | Region-tied bearer key from the turbopuffer dashboard. |
| `TURBOPUFFER_REGION` | No (default `gcp-us-central1`) | Part of the request host (`https://{region}.turbopuffer.com`); **must match the key's region**. |
| `TURBOPUFFER_NAMESPACE` | No (default `retrieval-bridge-tickets`) | The **pre-seeded** namespace the connector reads. |

> **Zero-key alternative for a quick local poke:** you can set
> `RETRIEVAL_BRIDGE_BACKEND=lancedb` and skip the turbopuffer keys, but only if you
> have a way to put a seeded LanceDB file where the container reads it. The
> headline, reproducible path is turbopuffer against a pre-seeded namespace.

You'll also need an **Anthropic key** for PromptQL itself (step 8); set that as a
*supergraph* env var (in the project's root `.env`), not a connector env var:

```bash
# in the project's root .env (NOT the connector .env):
ANTHROPIC_API_KEY=sk-ant-...
```

---

## 5. Introspect, add the Command, and paste the rich description

Introspect the connector so DDN reads its NDC schema (this needs the connector
buildable — `requirements.txt` resolvable). Then generate the Command HML and the
`SearchHit` ObjectType:

```bash
ddn connector introspect search
ddn command add search '*'
```

This writes (typically under `app/metadata/`):

- `SearchHit.hml` — the named ObjectType, auto-generated from the connector return
  type. **Let the CLI own this**; reference `ddn/metadata/SearchHit.hml` only to
  confirm the field set (it must include `id`).
- `search_documents.hml` — the Command, with a **draft `description` taken from the
  Python docstring**.

### Replace the draft description with the hand-tuned one

This is the most important manual edit in the whole runbook. **Tool selection in
PromptQL is driven by the metadata `description` field, not the code docstring.** A
vague description means the planner won't reach for hybrid retrieval when a user says
"find tickets like this". Open the generated `app/metadata/search_documents.hml` and
replace its `definition.description` (and confirm the argument descriptions) with the
rich version from **`ddn/metadata/search_documents.hml`** in this repo.

The description to paste (drives planner tool-selection):

```yaml
  description: |
    Hybrid (semantic + exact-keyword) search over the support-ticket corpus.

    USE THIS to find tickets relevant to a natural-language description of an
    incident, question, or symptom — e.g. "past resolved incidents that look
    like a 503 storm after a deploy", "tickets mentioning ERR_DIM_384", or
    "anything about the connection pool on the scale plan". It is the right tool
    whenever the user wants to retrieve, look up, find similar, or gather past
    tickets before you reason over them.

    It runs BOTH a vector (semantic-meaning) sub-query AND a BM25 (exact-token)
    sub-query, then fuses them with reciprocal-rank fusion. That hybrid is the
    point: pure vector search misses exact tokens like error codes (ERR_DIM_384,
    CERT_EXPIRED), ticket IDs (TCK-10042), and plan tiers, while BM25 misses
    paraphrases — together they catch both. Prefer this over a plain text scan.

    Returns the top_k most relevant tickets as SearchHit rows (best first by the
    fused `score`). Each SearchHit carries flat, filterable attributes (status,
    plan_tier, error_code, severity, root_cause, component, created_at) and an
    `id` (the ticket id). Follow the `account` relationship on a SearchHit to
    join its structured account facts (project_name, plan_tier, monthly_revenue,
    region, seat_count) — use that to rank or prioritise hits by customer value.

    Typical plan: search_documents(...) to narrow the corpus to a handful of
    candidates, then classify/summarize each, join `account`, and rank.
```

Keep the rest of the generated file (`outputType: "[SearchHit!]!"`, the four
`arguments` — `query`, `top_k`, `plan_tier`, `status` — and the `source` block
pointing `dataConnectorName: search` / `function: search_documents`). Use
`ddn/metadata/search_documents.hml` as the canonical reference for the argument
descriptions too.

> **Re-introspect after code changes.** If you later edit `functions.py` (signature
> or return shape), re-run `ddn connector introspect search` and rebuild — but
> re-applying that will regenerate the draft description, so **re-paste the rich
> description** above each time, or keep it in version control and re-copy.

---

## 6. Add Postgres (the Accounts model)

The join target — per-ticket account facts (project, plan tier, MRR, region, seats) —
lives in a Postgres table. This is the structured-facts side of the demo's "join
account facts by id" step.

### a. Create the table and load the rows

The DDL is `ddn/metadata/accounts.sql`. Create a database and run it:

```bash
createdb retrieval_bridge
psql retrieval_bridge -f "$RB_REPO/ddn/metadata/accounts.sql"
```

Load one row per ticket from the `"account"` object on each line of
`data/tickets.jsonl` (the **same** rows `scripts/seed.py` upserts into the local
SQLite store). A `jq` + `COPY` one-liner (any loader works):

```bash
jq -c '.account' "$RB_REPO/data/tickets.jsonl" \
  | jq -r '[.ticket_id,.project_id,.project_name,.plan_tier,
            .monthly_revenue,.account_region,.seat_count] | @csv' \
  | psql retrieval_bridge -c \
      "COPY accounts (ticket_id, project_id, project_name, plan_tier,
                      monthly_revenue, account_region, seat_count)
       FROM STDIN WITH (FORMAT csv)"
```

The `accounts` table is keyed by **`ticket_id`** — that is the join key
`SearchHit.id` maps to in step 7. The column set matches
`retrieval_bridge/structured.py` 1:1, so the deployed Postgres join behaves
identically to the local SQLite analog.

### b. Scaffold the Postgres connector and introspect

```bash
ddn connector init pg -i        # choose hasura/postgres; point it at your database
ddn connector introspect pg
```

The Postgres connector config is **Version 5**. If your database isn't reachable from
the container on `localhost`, use a host-reachable URL (on Docker Desktop,
`host.docker.internal` instead of `localhost`). Register the connection string with
`ddn connector env add pg --env ...` as the init prompts indicate.

### c. Add the model

```bash
ddn model add pg accounts
```

This introspects the table and emits three blocks — the `Accounts` ObjectType, the
`Accounts` Model, and its ModelPermissions (plus filter/aggregate boilerplate). Let
the CLI generate these; `ddn/metadata/accounts.hml` is a committed, hand-trimmed
reference so you can read the join end-to-end, but on a live project you let
`ddn model add` produce it from the table.

> **Foreign-key relationships (optional, Postgres-only).** If you had multiple
> related Postgres tables, `ddn relationship add pg '*'` would generate
> relationships from their FKs. We don't need that here — there is only one table.
> The relationship we *do* need (next step) is **not** a Postgres FK and
> `ddn relationship add` will **not** create it.

---

## 7. Hand-add the SearchHit → Accounts relationship

This is the "join structured facts to a hit, with no join code" claim, expressed
declaratively. It is the **one HML file you author by hand.**

> **`ddn relationship add` will NOT generate this.** That command only generates
> relationships from **Postgres foreign keys between two Postgres Models**. Here the
> source is a **Command return type** (`SearchHit`, from the Python connector), not a
> Postgres model, and there is no FK. So you write the file yourself.

Create `app/metadata/relationship_searchhit_account.hml` (place it in the same
subgraph metadata folder as the others) with the contents of
**`ddn/metadata/relationship_searchhit_account.hml`**:

```yaml
kind: Relationship
version: v1
definition:
  name: account

  # Source is the COMMAND RETURN TYPE, not the command. (SearchHit.)
  sourceType: SearchHit

  target:
    model:
      name: Accounts
      subgraph: app           # the subgraph that owns the Accounts model
      relationshipType: Object   # REQUIRED for a single-row model target

  # Join key mapping: SearchHit.id  →  Accounts.ticket_id.
  mapping:
    - source:
        fieldPath:
          - fieldName: id
      target:
        modelField:
          - fieldName: ticket_id

  description: |
    The account facts for this retrieved ticket. Resolves SearchHit.id against
    Accounts.ticket_id so a plan can join project_name / plan_tier /
    monthly_revenue to each hit and rank or prioritise by customer value —
    declaratively, with no join code.
```

Three things that bite people if you get them wrong:

- **`sourceType` is the RETURN TYPE NAME (`SearchHit`), never the command name**
  (`search_documents`). Relationships attach to a *type*. Naming `account` on
  `SearchHit` is what makes `account { ... }` a legal selection on a hit.
- **`relationshipType: Object` is REQUIRED** for a model target that resolves to one
  row per source (`ticket_id` is unique). Use `Array` only for one-to-many.
- **`subgraph: app`** must match the subgraph that actually owns the `Accounts` model
  from step 6. If you used a different subgraph name in step 2, change it here.

After this exists, a plan (or raw GraphQL) can ask:

```graphql
search_documents(query: "503 after deploy") {
  id  score  text
  account { project_name  plan_tier  monthly_revenue }   # ← resolved by the relationship
}
```

---

## 8. Configure PromptQL

PromptQL is turned on by a single global config: `PromptQlConfig` (kind:
`PromptQlConfig`, version: **v2**) at **`globals/metadata/promptql-config.hml`**.
`ddn supergraph init --with-promptql` (step 2) scaffolds one; replace/merge its
contents with **`ddn/globals/promptql-config.hml`** from this repo.

Key fields:

- **`llm`** — the model that drives **planning** (turning a question into a query
  plan). Anthropic is the recommended provider.
- **`aiPrimitivesLlm`** — the model backing the **AI primitives** the plan calls.
  PromptQL's primitives are exactly **`summarize`, `classify`, `extract`,
  `visualize`** (invoked as `executor.<name>(...)`). Our demo uses `classify` (root
  cause) and `summarize` (the fix). `aiPrimitivesLlm` can match `llm` or differ; an
  optional `overrideAiPrimitivesLlm` exists for finer control.
- **`apiKey.valueFromEnv: ANTHROPIC_API_KEY`** — never hard-code a key; it's read
  from the env you set in step 4 (`ANTHROPIC_API_KEY` as a supergraph env var).
- **`systemInstructions`** — planner steering that reinforces the `description`
  fields: always `search_documents` first for incident/lookup questions, include
  literal error codes/IDs verbatim in `query`, keep `top_k` small (5–10), follow the
  `account` relationship for customer context, and use the primitives only where
  judgment is needed.

> **Model strings are point-in-time.** The committed config uses
> `claude-sonnet-4-5-20250929`, which was current when it was written and **will
> drift**. Before you build, replace it with a current Anthropic model id from the
> Anthropic docs (both `llm.model` and `aiPrimitivesLlm.model`).

---

## 9. Build, run, and open the Playground

```bash
ddn supergraph build local       # compiles all the HML into a local supergraph build
ddn run docker-start             # starts the engine + connectors + Postgres in Docker
ddn console --local              # opens the PromptQL Playground for the local build
```

- `ddn supergraph build local` is what surfaces config errors — a bad `subgraph:`,
  a missing `id` field on `SearchHit`, a malformed relationship mapping. Fix and
  rebuild until it's clean.
- `ddn run docker-start` needs the Compose v2.20+ from step 0. This is also where the
  connector container installs `requirements.txt` (the `git+https` retrieval-bridge
  package) — the first start is slower while that resolves.
- `ddn console --local` opens the **PromptQL Playground** for your local build at
  `promptql.console.hasura.io` (pointed at your local engine). This is where you ask
  the demo prompt.

If you re-edit any `functions.py` or HML after this, re-run
`ddn supergraph build local` (and re-introspect the connector if the function
signature/return type changed), then refresh the console.

---

## 10. Ask the demo prompt

In the PromptQL Playground (`ddn console --local`), ask the incident question the
local `scripts/demo.py` is built around:

> We just had a production incident: right after the 14:00 deploy, our search API
> started returning 503 Service Unavailable and requests timed out under heavy load.
> EU customers were hit hardest. Which past resolved incidents look like this, what
> was the root cause of each, and how were they fixed?

What a correct plan does (the same five-step division of labor the local demo runs):

1. **`search_documents(query=<incident>, top_k≈8, status="resolved")`** — hybrid
   retrieval. The 503 / timeout tokens are matched exactly by BM25 *and* semantically
   related tickets (e.g. DB-pool exhaustion) surface via the vector sub-query, fused
   with RRF. In the local run this returns **7 exact-503 tickets + 1 semantically
   related pool-exhaustion ticket**.
2. **`classify`** the root cause of each hit (PromptQL primitive).
3. **`summarize`** how each was fixed (PromptQL primitive).
4. **Follow the `account` relationship** on each `SearchHit` to join project / plan /
   MRR — no join code, resolved by the relationship from step 7.
5. **Rank** by plan tier, then recency (deterministic) — scale-tier accounts get
   promoted above launch.

If the planner *doesn't* call `search_documents`, the usual cause is a weak Command
`description` (step 5) or missing `systemInstructions` (step 8) — both steer tool
selection. Re-check those, rebuild, and ask again.

### Why hybrid (the measured payoff)

From `scripts/bench.py` against this corpus:

- **Exact-id lookup (Section A):** pure-vector **MRR 0.117**, found@5 = **1/12**;
  **hybrid MRR 0.572**, found@5 = **12/12**. Exact tokens are where pure vector
  search falls down and BM25 carries it.
- **Semantic queries (Section B):** vector and hybrid both **5/5 hits@5** — hybrid
  **does not regress** on the meaning-based queries. You get the exact-token wins for
  free.

---

## Appendix: connect to an existing cloud PromptQL project

If you already have a **browser PromptQL project** and want the same `search_documents`
command + `account` join available there (not just in the local build):

1. **Use the same Hasura account.** `ddn auth login` (step 1) with the account that
   owns the browser project so the CLI can see it.
2. **Point the local checkout at the cloud project.** Either initialize the
   supergraph inside the existing project context (the CLI associates the build with a
   project id), or link your local checkout to it. `ddn project` subcommands list and
   select the project; follow the CLI prompts so builds target the right project.
3. **Build to the cloud, not local.** Instead of `ddn supergraph build local`, run a
   **cloud build** (`ddn supergraph build create`). That produces a hosted build the
   browser PromptQL Playground can run against.
4. **Set secrets in the cloud project, not in local `.env`.** The turbopuffer keys
   (step 4) and `ANTHROPIC_API_KEY` (step 8) must exist as **environment
   variables/secrets in the cloud project**, since the hosted connector reads them
   there. Add them via the CLI's env/secrets commands or the project settings UI.
5. **Open the cloud Playground** from the browser PromptQL project (or
   `ddn console` without `--local`). Ask the [demo prompt](#10-ask-the-demo-prompt).

Ways to *drive* PromptQL once it's wired in (all equivalent on the same project):

- **The Playground** — `ddn console --local`, or the cloud console.
- **The Natural Language API** — `POST https://promptql.ddn.hasura.app/api/query`
  with a `Bearer` **PromptQL API key** (from project settings).
- **The Python SDK** — `promptql-api-sdk`.

There is **no `promptql` CLI binary** — the CLI is always **`ddn`**; PromptQL is the
config + Playground/API on top of the DDN supergraph you built above.

---

### Related runbooks

- **[turbopuffer-runbook.md](turbopuffer-runbook.md)** — move the demo from local
  LanceDB onto a real turbopuffer namespace (one extra, three env vars). Seed the
  namespace there before pointing the deployed connector at it.
- **[pitch.md](pitch.md)** — the why: turbopuffer as a swappable hybrid-retrieval
  backend for PromptQL, and the architecture this runbook deploys.
