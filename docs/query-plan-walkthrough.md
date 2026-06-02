# Query-plan walkthrough

What PromptQL is **expected to do** with the demo prompt, annotated step by step —
and how each step maps to where it runs in this repo. The query plan below is the
deployed-system twin of what `scripts/demo.py` already executes locally and
zero-key; the numbers cited are the **real demo output** (LanceDB backend,
`bge-small-en-v1.5` embedder).

> The point of this doc is the **shape** of the plan — retrieve → classify →
> summarize → join → rank — and the **seam** the bridge sits in. The backend
> (turbopuffer headline / LanceDB clone-and-run default / pgvector optional) is
> swappable underneath the `search_documents` command without changing a single
> plan step.

---

## 1. The prompt

Paste this into the **PromptQL Playground** (the project must already have the
`search_documents` command, the `Accounts` model, and the `SearchHit → Accounts`
relationship in its metadata — see [`docs/ddn-runbook.md`](ddn-runbook.md)):

```
Find the 8 past resolved tickets most similar to this new outage report:

"We just had a production incident: right after the 14:00 deploy, our search API
started returning 503 Service Unavailable and requests timed out under heavy
load. EU customers were hit hardest."

Classify each root cause, summarize the fix, then rank by plan tier and recency.
```

This is the same incident text `scripts/demo.py` hard-codes as `INCIDENT`. The
phrasing is deliberate: it is natural language (no SQL, no filter syntax), it
names a number of results (`8`), an exact status (`resolved`), and three reasoning
verbs (`classify`, `summarize`, `rank`) that the planner must turn into a
sequence of `executor.*` calls plus a deterministic sort.

---

## 2. The query plan PromptQL is expected to generate

PromptQL emits a deterministic Python-style plan, then executes it inside its
runtime. Annotated pseudo-Python — the calls and their order are what to expect:

```python
# ── Step 1: HYBRID RETRIEVAL (the bridge) ────────────────────────────────────
# The planner reads the search_documents Command `description` and picks it for
# "find … most similar to this outage report". It maps "8" -> top_k, the
# natural-language outage text -> query (verbatim, so BM25 can match the literal
# "503"), and "past resolved" -> the status filter.
results = search_documents(
    query="We just had a production incident: right after the 14:00 deploy, our "
          "search API started returning 503 Service Unavailable and requests "
          "timed out under heavy load. EU customers were hit hardest.",
    top_k=8,
    status="resolved",
)

# The retrieved hits are STORED AS AN ARTIFACT — a typed table PromptQL holds in
# its runtime and reasons over deterministically, not re-derived per step.
incident_candidates = as_artifact(results)        # rows of SearchHit

# ── Step 2: CLASSIFY each root cause (PromptQL primitive) ─────────────────────
# executor.classify runs an LLM call in isolated, focused context per hit (or
# batched over the artifact). Output is a categorical label per row.
for hit in incident_candidates:
    hit.root_cause_label = executor.classify(
        hit.text,
        instructions="Classify the root cause of this incident",
        # categories the planner may enumerate, e.g.:
        categories=["upstream_overload", "pool_exhausted", "out_of_memory",
                    "dimension_mismatch", "rate_limited", ...],
    )

# ── Step 3: SUMMARIZE the fix (PromptQL primitive) ───────────────────────────
for hit in incident_candidates:
    hit.fix_summary = executor.summarize(
        hit.text,
        instructions="Summarize how this incident was fixed (the resolution).",
    )

# ── Step 4: JOIN account facts (DDN relationship, no join code) ───────────────
# `account` is the hand-authored relationship SearchHit.id -> Accounts.ticket_id.
# The planner traverses it to pull project_name / plan_tier / monthly_revenue.
for hit in incident_candidates:
    acct = hit.account                      # -> Accounts row via the relationship
    hit.plan_tier       = acct.plan_tier
    hit.monthly_revenue = acct.monthly_revenue
    hit.project_name    = acct.project_name

# ── Step 5: RANK deterministically (plain code, not an LLM) ───────────────────
PLAN_RANK = {"enterprise": 3, "scale": 2, "launch": 1, "free": 0}
ranked = sorted(
    incident_candidates,
    key=lambda h: (PLAN_RANK[h.plan_tier], h.created_at),   # tier, then recency
    reverse=True,                                            # highest tier, newest first
)
return ranked                                # the final, ordered answer
```

Notes on what makes this plan correct:

- **`search_documents` is the only retrieval step.** There is no second scan, no
  raw GraphQL `SELECT` over text. Hybrid retrieval narrows 160 tickets (millions
  in production) to 8 candidates in one call.
- **`status="resolved"` is a `search_documents` argument**, not a post-filter —
  the planner pushes the facet into the retrieval call (the connector renders it
  into the backend's own filter DSL: turbopuffer tuple filters, LanceDB SQL,
  pgvector `WHERE`).
- **The hits become an artifact** before any reasoning. classify/summarize/rank
  all operate over that one stored table — they never silently re-query the
  vector store.
- **classify and summarize are `executor.*` primitives** (LLM-backed, focused
  context), not free-form generation. Ranking is deterministic Python, not an
  LLM — so the order is reproducible.
- **The join is a relationship traversal (`hit.account`)**, not application code:
  PromptQL follows the declarative `SearchHit → Accounts` edge.

---

## 3. Where each plan step runs (local demo vs. deployed component)

| # | Plan step | Local `scripts/demo.py` equivalent | Deployed component |
|---|-----------|------------------------------------|--------------------|
| 1 | `search_documents(query, top_k=8, status='resolved')` → hybrid vector + BM25, fused with RRF | `bridge.search_documents(INCIDENT, top_k=8, filters={"status": "resolved"})` (`demo.py:69`) → `RetrievalBridge.search_documents` over the seeded backend | **DDN `search_documents` Command** (`ddn/metadata/search_documents.hml`) → **Python lambda connector** (`ddn/connector/search/functions.py:83`) → **`RetrievalBridge`** → **turbopuffer / LanceDB / pgvector** (`RETRIEVAL_BRIDGE_BACKEND`) |
| — | hits **stored as an artifact** | the `hits` / `enriched` list held in memory across steps (`demo.py:69`, `79`) | PromptQL **artifact** in its deterministic runtime |
| 2 | `executor.classify(...)` → root cause per hit | `classify_root_cause(h.text)` (`demo.py:82`) — deterministic stand-in in `retrieval_bridge/demo_primitives.py` | **PromptQL `executor.classify`** primitive (LLM in focused context), model from `PromptQlConfig.aiPrimitivesLlm` |
| 3 | `executor.summarize(...)` → the fix per hit | `summarize_fix(h.text)` (`demo.py:83`) — deterministic stand-in in `retrieval_bridge/demo_primitives.py` | **PromptQL `executor.summarize`** primitive |
| 4 | join account facts by `id` (`hit.account`) | `store.get_by_ids([h.id for h in hits])` (`demo.py:80`) — `StructuredStore` (SQLite) keyed by ticket id | **DDN relationship** `SearchHit → Accounts` (`relationship_searchhit_account.hml`) → **Postgres `Accounts` model** (`ddn/metadata/accounts.hml`) |
| 5 | rank by plan tier, then recency | `enriched.sort(key=lambda e: (PLAN_RANK[...], e["created_at"]), reverse=True)` (`demo.py:110`) | PromptQL **deterministic plan code** (plain sort, not an LLM call) |

The `*` and `**` in `demo.py`'s header comment flag exactly these two
emulations: classify/summarize are emulated by `demo_primitives`, and the
relationship join is emulated by `StructuredStore.get_by_ids` (the SQLite
Postgres analog). **Step 1 is not emulated** — the demo runs the identical bridge
code the deployed connector calls, which is the whole swappable-backend pitch.

### The five PromptQL/DDN equivalences

| Plan element | Deployed (real) | Demo (local) |
|--------------|-----------------|--------------|
| `search_documents` | DDN Command over the lambda connector → `RetrievalBridge` (real, unchanged code) | `bridge.search_documents(...)` |
| `executor.classify` | PromptQL primitive (LLM) | `demo_primitives.classify_root_cause` |
| `executor.summarize` | PromptQL primitive (LLM) | `demo_primitives.summarize_fix` |
| `hit.account` join | DDN relationship `SearchHit.id → Accounts.ticket_id` | `StructuredStore.get_by_ids` (SQLite) |
| rank by tier, recency | deterministic plan code | `enriched.sort(...)` in `demo.py` |

---

## 4. What to verify in the generated plan (is the bridge actually being used?)

When you run the prompt, expand the generated plan in the Playground and confirm
all three of these. If any is missing, PromptQL is **not** routing through the
turbopuffer/LanceDB bridge:

1. **A `search_documents(...)` call is the retrieval step.**
   - The plan must call `search_documents` with `query` set to the outage text
     **verbatim** (so BM25 can match the literal token `503`), `top_k=8`, and
     `status="resolved"`. If you instead see a plain GraphQL `accounts` query, a
     raw text scan, or the model answering from its own memory, the bridge is
     bypassed. The Command `description` in `search_documents.hml` is what makes
     the planner reach for this tool — that text, not the Python docstring, drives
     tool selection.

2. **The hits are stored as an artifact before any reasoning.**
   - The plan should capture the `search_documents` result into a named artifact
     (a typed `SearchHit` table) and then run classify / summarize / rank over
     **that artifact**. classify and summarize must consume the stored hits — they
     must **not** re-issue retrieval or invent ticket content. One retrieval call,
     N reasoning calls over the same stored rows.

3. **The account join is a relationship traversal (`hit.account`), not a second query.**
   - The plan must reach account facts (`project_name`, `plan_tier`,
     `monthly_revenue`) by following the `account` relationship off each
     `SearchHit` (`SearchHit.id → Accounts.ticket_id`) — not by a separately
     planned, manually-joined `accounts` lookup. This is what proves the
     declarative DDN relationship is wired (`relationshipType: Object`,
     `sourceType: SearchHit`, mapping `id → ticket_id`). The deterministic sort
     keys (plan tier, then recency) must come from these joined fields.

A fourth, backend-side check (outside the plan view): the connector reads
`RETRIEVAL_BRIDGE_BACKEND`. With it set to `turbopuffer`, a single
`search_documents` turn issues a vector-ANN sub-query and a BM25 sub-query fused
client-side with RRF (turbopuffer has no server-side fusion, so RRF stays in the
bridge). Warm turbopuffer reads are ~14 ms, cold reads off object storage a few
hundred ms — well inside one interactive PromptQL turn.

---

## 5. The expected result (real demo output)

This is the **actual** output of `python scripts/demo.py` (LanceDB +
`bge-small-en-v1.5`). The deployed plan over turbopuffer produces the same shape;
absolute fused scores are ranking-only and not comparable across backends.

**Step 1 — `search_documents()` returned 8 candidates (fused RRF, best first):**

```
TCK-10159  score=0.0325  [ERR_GW_503]  Intermittent 503 Service Unavailable on search …
TCK-10114  score=0.0325  [ERR_GW_503]  Production API returning 503 errors under load
TCK-10068  score=0.0315  [ERR_GW_503]  Intermittent 503 Service Unavailable on search …
TCK-10108  score=0.0310  [ERR_GW_503]  Production API returning 503 errors under load
TCK-10071  score=0.0308  [ERR_GW_503]  Gateway throwing 503s after traffic spike
TCK-10069  score=0.0308  [HTTP 503]    Production API returning 503 errors under load
TCK-10060  score=0.0292  [ERR_GW_503]  Intermittent 503 Service Unavailable on search …
TCK-10127  score=0.0288  [ERR_POOL_53] 500 errors traced to exhausted DB pool
```

This is the hybrid payoff in one screen: **7 exact-503 tickets surfaced by BM25**
(the literal `503` / `ERR_GW_503` token) **plus 1 semantically-related
DB-pool-exhaustion ticket** (`TCK-10127`) that vector search pulled in because it
is conceptually the same class of "overload under load" incident even though it
never contains the string `503`. Pure vector search would miss the exact-token
matches; pure keyword search would miss `TCK-10127`.

**Steps 2–4 — classify root cause, summarize fix, join account facts.** Every
classify output matched the ticket's ground-truth `root_cause` label (the `✓`
column): seven `upstream_overload`, one `pool_exhausted`. Each hit's `id` joined
to its account row (project / plan / MRR).

**Step 5 — final answer, ranked by plan tier then recency:**

```
#  ticket     plan          MRR  opened      root cause          project
1  TCK-10069  scale     $12,400  2025-05-19  upstream_overload   Pavo Retail
2  TCK-10060  scale     $12,400  2025-02-25  upstream_overload   Pavo Retail
3  TCK-10127  scale     $ 7,200  2025-01-19  pool_exhausted      Lyra Media
4  TCK-10108  launch    $   640  2026-05-21  upstream_overload   Draco Games
5  TCK-10068  launch    $ 1,100  2026-05-05  upstream_overload   Nova Robotics
6  TCK-10071  launch    $   640  2026-04-29  upstream_overload   Draco Games
7  TCK-10114  launch    $ 1,100  2025-12-05  upstream_overload   Nova Robotics
8  TCK-10159  launch    $   640  2025-07-05  upstream_overload   Draco Games
```

The ranking shows why the join matters: the three **scale-tier** accounts
(Pavo Retail $12.4k, $12.4k; Lyra Media $7.2k MRR) are **promoted above every
launch-tier** ticket regardless of retrieval score, and within a tier newer
incidents come first. The planner did not need a higher-scored ticket to be a
higher-priority answer — customer value (a structured fact joined from Postgres)
drives the final order, while hybrid retrieval found the right candidates in the
first place.

---

## See also

- [`docs/ddn-runbook.md`](ddn-runbook.md) — register the command, model, and the
  `SearchHit → Accounts` relationship; point your PromptQL project at it.
- [`docs/turbopuffer-runbook.md`](turbopuffer-runbook.md) — seed a real
  turbopuffer namespace and flip `RETRIEVAL_BRIDGE_BACKEND`.
- `scripts/demo.py` — runs this exact plan offline; `scripts/bench.py` — the
  hybrid-vs-vector measurements behind the pitch.
