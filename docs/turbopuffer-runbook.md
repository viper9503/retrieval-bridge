# turbopuffer runbook — move the demo from local LanceDB onto real turbopuffer

This is the one-time upgrade path. The local demo (LanceDB + fastembed) runs with
**zero API keys** — that is the on-ramp. When you want to see the *same* query
plan running against turbopuffer's serverless hybrid store, follow the steps
below. The headline thing to keep in mind:

> **Nothing about the commands or the query plan changes.** You install one extra,
> set three env vars, and re-run the exact same `seed.py` / `demo.py` / `bench.py`.
> The backend is swapped behind the `VectorBackend` seam — the PromptQL-facing
> `search_documents` command never moves.

---

## 0. Heads-up on cost (read this first)

turbopuffer is priced for production scale, not for a free playground. Be upfront
with yourself before you commit:

- **No free tier.** The lowest plan is **Launch = $64/month minimum** — you pay the
  greater of metered usage or $64. (The next tier, Scale, is $256/mo.)
- A few-thousand-vector demo like this one is *far* under the technical minimum,
  but you still commit to the **$64/mo Launch floor** just to get a key.
- There is a **30-day cancel-for-refund window**, so a short evaluation is
  refundable if you cancel inside it. Set yourself a reminder.

This is exactly *why* this repo defaults to local LanceDB: the demo is free, and
turbopuffer is a one-env-var upgrade. None of that is a knock on turbopuffer — it
is the open-source-analog (LanceDB) doing the on-ramp, and turbopuffer doing the
production-scale headline. If you only want to see the architecture work, **stay on
LanceDB**; you do not need any of the steps below.

---

## 1. Sign up

1. Go to **https://turbopuffer.com/join**.
2. Sign up with your email and pick a plan. Self-serve signup is immediate —
   **no waitlist, no approval gate**. (You are accepting the $64/mo Launch minimum
   noted above.)

---

## 2. Choose a region — and know the key is region-tied

turbopuffer is **region-scoped**: the region is part of the request host
(`https://{region}.turbopuffer.com`), and **your API key is tied to the region you
create it in**. There is no global endpoint — the legacy `api.turbopuffer.com` is
dead, do not use it.

- Pick a region close to where you'll run the demo, e.g. `gcp-us-central1` or
  `aws-us-east-1`.
- Whatever you choose here must match `TURBOPUFFER_REGION` in step 4. A mismatch
  between the key's region and the env var will fail to authenticate.
- The repo defaults to `gcp-us-central1` (same region the live docs use), so if you
  pick that one you can skip setting `TURBOPUFFER_REGION` entirely.

---

## 3. Create an API key

1. Open the dashboard at **https://turbopuffer.com/dashboard**.
2. Create an API key (in your chosen region) and copy it. You won't see it again.
3. Auth is a bearer token — the SDK sends `Authorization: Bearer <key>` for you;
   you just hand it the key via env.

---

## 4. Install the extra and set the environment

From the repo root, with your virtualenv active:

```bash
pip install -e '.[turbopuffer]'        # adds the turbopuffer v2 SDK
```

The turbopuffer SDK is an **optional extra** — the base install never pulls it in,
and the backend imports it lazily, so this step is the only thing that adds it.

Then set the three environment variables. The simplest path is to copy the example
file and fill it in:

```bash
cp .env.example .env
# edit .env and set TURBOPUFFER_API_KEY (and TURBOPUFFER_REGION if not gcp-us-central1)
```

Or export them directly in your shell:

```bash
export TURBOPUFFER_API_KEY=tpuf_...                 # from step 3 (required)
export TURBOPUFFER_REGION=gcp-us-central1           # must match the key's region
export TURBOPUFFER_NAMESPACE=retrieval-bridge-tickets   # optional; this is the default
```

| Env var | Required? | Default | Notes |
|---|---|---|---|
| `TURBOPUFFER_API_KEY` | **Yes** | — | Region-tied key from the dashboard. Seeding errors out with a clear message if it's unset. |
| `TURBOPUFFER_REGION` | No | `gcp-us-central1` | Part of the host. Must match the key's region. |
| `TURBOPUFFER_NAMESPACE` | No | `retrieval-bridge-tickets` | The namespace to write/query. **Created implicitly on first write** — you don't pre-create it. |

You also flip the backend selector — but you do that per-command (next section), not
in `.env`, so the same checkout still runs the free local demo by default.

---

## 5. Seed turbopuffer

Point the backend at turbopuffer for the seed run. The corpus, the embedder, and
the command are all unchanged — only where the vectors land changes.

```bash
RETRIEVAL_BRIDGE_BACKEND=turbopuffer python scripts/seed.py
```

What this does:

- Embeds the 160 synthetic tickets locally with fastembed BGE-small (384-dim) — the
  **same** default embedder as the local run. (Set `RETRIEVAL_BRIDGE_EMBEDDER` only
  if you want a cloud embedder.)
- Writes them to your namespace with the hybrid schema turbopuffer needs: a
  `[384]f32` vector column with `ann: true`, and the `text` column flagged
  `full_text_search` (BM25 requires that flag). Every other attribute
  (`status`, `plan_tier`, `error_code`, etc.) is filterable by default.
- Writes the same structured "account" rows into the local SQLite store (the DDN
  Postgres-join analog) — that part is unchanged from the local run.

The namespace appears in your dashboard after this first write. Expected output ends
with `Indexed 160 documents into turbopuffer.`

---

## 6. Run the demo and the benchmark against turbopuffer

```bash
RETRIEVAL_BRIDGE_BACKEND=turbopuffer python scripts/demo.py
RETRIEVAL_BRIDGE_BACKEND=turbopuffer python scripts/bench.py
```

The command and the plan are **identical to the local run** — compare to the
LanceDB version:

```bash
# local (free) — what you ran first:
python scripts/seed.py && python scripts/demo.py

# turbopuffer — same scripts, same flags, one env var:
RETRIEVAL_BRIDGE_BACKEND=turbopuffer python scripts/seed.py && \
RETRIEVAL_BRIDGE_BACKEND=turbopuffer python scripts/demo.py
```

`demo.py` prints the backend it's using in its `STEP 1 — search_documents()`
header, so you can confirm it says `backend: turbopuffer`. The end-to-end plan is
the same five steps regardless of backend:

1. `search_documents(query=<incident>, top_k=8, filters={status:'resolved'})` — now
   served by turbopuffer as one snapshot-isolated `multi_query` (vector ANN +
   BM25), fused **client-side** with reciprocal-rank fusion because turbopuffer has
   no server-side fusion.
2. classify root cause (PromptQL `executor.classify`, emulated locally).
3. summarize the fix (PromptQL `executor.summarize`, emulated locally).
4. join account facts by id (DDN relationship → Postgres model, emulated by SQLite).
5. rank by plan tier, then recency (deterministic Python).

Steps 2–5 don't touch the vector store at all, so they behave identically. Only
step 1's two sub-queries now run in turbopuffer.

### Makefile shortcut

The convenience targets respect the same env var:

```bash
RETRIEVAL_BRIDGE_BACKEND=turbopuffer make seed
RETRIEVAL_BRIDGE_BACKEND=turbopuffer make demo
RETRIEVAL_BRIDGE_BACKEND=turbopuffer make bench
```

---

## 7. Cold vs. warm latency (and pre-warming)

turbopuffer is object-storage-native and tiered, so the **first** query against a
namespace that hasn't been touched recently is a *cold* read straight from object
storage — expect **hundreds of milliseconds**. Once the namespace is warm it serves
in **~14 ms**.

For a demo this matters in one place: the very first `search_documents` of a session
can look slow, then everything after it is fast. Two ways to handle it:

- **Pre-warm before you present.** Run `demo.py` (or `bench.py`) once right before
  you show it — that first query pulls the namespace warm, and the run you actually
  demo is the fast one. `bench.py` re-queries the namespace many times, so it warms
  itself after the first lookup.
- **Don't read cold latency as a turbopuffer weakness.** It's the cost model: you're
  paying object-storage prices (~10× cheaper than RAM-resident vector DBs) and the
  warm path is the steady state. Mention it, then show the warm numbers.

---

## 8. Go back to the free local demo anytime

Because the backend is selected per-command, you never have to "undo" anything:

```bash
python scripts/demo.py            # LanceDB again — the default, zero keys
```

If you finished evaluating and want to stop billing, **cancel inside the 30-day
window** for a refund (step 0). The local LanceDB demo keeps working with no key,
forever.

---

## What stayed the same (the whole point)

| | Local (LanceDB) | turbopuffer |
|---|---|---|
| Command to seed | `python scripts/seed.py` | same + `RETRIEVAL_BRIDGE_BACKEND=turbopuffer` |
| Command to demo | `python scripts/demo.py` | same + `RETRIEVAL_BRIDGE_BACKEND=turbopuffer` |
| Embedder | fastembed BGE-small (384-dim) | same (local default) |
| `search_documents` signature | `search_documents(query, top_k, filters)` | **identical** |
| Hybrid recipe | vector + FTS + client-side RRF | vector + BM25 multi_query + client-side RRF |
| The 5-step query plan | unchanged | unchanged |

One extra, three env vars, one selector flag. That's the upgrade.

---

### Related runbooks

- **[ddn-runbook.md](ddn-runbook.md)** — deploy the lambda connector and register
  `search_documents` in your PromptQL/DDN project.
- **[verified-facts.md](verified-facts.md)** — the live-docs citations behind the
  turbopuffer v2 API, region-scoped hosts, and pricing used here.
