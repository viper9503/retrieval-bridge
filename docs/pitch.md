# The pitch: turbopuffer as a swappable hybrid-retrieval backend for PromptQL

**One natural-language question → a deterministic PromptQL query plan → a single `search_documents` command → hybrid (vector + keyword) retrieval → classify, summarize, join structured facts, rank. The retrieval backend behind that command is swappable — turbopuffer is the headline default, LanceDB is the free clone-and-run analog, pgvector is the DDN-native option — and none of them change a line of the PromptQL-facing interface.**

This document is the long version. If you want it in one paragraph, read the [README](../README.md). If you want to run it, the README quickstart takes about two minutes and needs no API keys. This is the *why*.

---

## The thesis

PromptQL is not "RAG with extra steps." It is three things working together:

1. **Planning** — an LLM turns a question into a *query plan that is a Python program*, not a single SQL string. The plan can branch, loop, and adapt mid-run.
2. **Deterministic execution** — the plan runs as code. Data lives in artifacts outside the LLM context window, so results are reproducible and auditable, not re-hallucinated on each turn.
3. **Reasoning primitives** — focused LLM calls (`classify` / `summarize` / `extract` / `visualize`) invoked *only* where judgment is actually required, on small inputs the plan already narrowed down.

That architecture has one structural dependency it does not solve on its own: **getting from millions of unstructured documents down to the handful worth reasoning over.** A plan can classify a root cause beautifully — but only if the right ten tickets are already in front of it. Retrieval is the funnel that feeds every reasoning primitive.

**turbopuffer is exactly that funnel.** It is serverless **hybrid** search — vector similarity for *meaning* plus BM25 for *exact tokens* (SKUs, error codes, order refs, IDs) — built on object storage instead of RAM. It narrows millions to dozens, cheaply, and it does the one thing pure-vector retrieval is structurally bad at: finding the literal token.

The two need a seam to meet at, and Hasura DDN already ships it: **the Python lambda connector.** A lambda connector turns an ordinary Python function into a first-class DDN *command*. So `search_documents` becomes a command PromptQL can plan against — with zero bespoke plumbing, no new protocol, no custom gateway. It is the existing extension point, used for exactly what it was designed for.

And because the command body sits behind a one-method backend interface (`VectorBackend.search`), turbopuffer is the **default, not a lock-in**. Swap the env var and the same command runs on LanceDB or pgvector. The plan, the command signature, and the reasoning steps are byte-for-byte identical. Only *where the two sub-queries execute* changes.

This repo proves all of that end to end — and runs with no API keys.

---

## The division of labor (who does what)

| Layer | Job | Why it's the right tool |
|---|---|---|
| **turbopuffer** | High-recall, low-cost **retrieval**: millions → dozens | Vector **+ BM25** hybrid fixes the pure-vector blind spot on exact tokens. Serverless on object storage → roughly 10x cheaper than RAM-resident vector DBs. |
| **PromptQL** | **Planning + deterministic execution + reasoning** | The plan is a Python program: it adapts mid-run, keeps artifacts outside the LLM context, and calls focused LLM primitives only where judgment is needed. |
| **Postgres via DDN** | **Structured facts** joined to retrieved candidates | A retrieved ticket id joins to its account row (plan tier, MRR) through a *declarative* DDN relationship — no join code. |

Retrieval is a **deterministic tool inside the plan**, not the answer. That is the framing PromptQL stands for, and turbopuffer slots into it cleanly.

---

## Why hybrid matters (and the numbers that prove it)

Pure-vector retrieval has a real, structural failure mode: **exact identifiers.** Embeddings map look-alike tokens — `ERR_DIM_384`, `ERR_DIM_768`, a ticket id, a plan name — into nearly the same region of vector space. They are *semantically* almost identical, so the model cannot tell them apart. The true match gets ranked low, or missed entirely.

BM25 (keyword) retrieval has the opposite strength: it pins the literal token. Hybrid runs both and fuses the rankings (reciprocal-rank fusion), so you get meaning *and* exactness. The question is whether adding BM25 helps on exact lookups without hurting semantic recall. The repo's `scripts/bench.py` measures exactly that, on the 160-ticket corpus:

### Section A — exact-identifier lookup (vector's blind spot)

| Mode | MRR | found@5 |
|---|---|---|
| Pure vector | 0.117 | **1 / 12** |
| **Hybrid** | **0.572** | **12 / 12** |

Looking up a specific id among look-alikes, pure vector found it in the top 5 **once out of twelve tries.** Hybrid found it **every single time** — and roughly 5x the MRR. This is the headline turbopuffer argument made concrete: the exact-token win is not marginal, it is the difference between a retrieval step that works and one that silently drops the document the user named.

### Section B — semantic query (vector's strength)

| Mode | hits@5 |
|---|---|
| Pure vector | 5 / 5 |
| **Hybrid** | **5 / 5** |

On paraphrased, token-free problems ("customers are being signed out unexpectedly", "the first search each morning is slow"), hybrid matches pure vector exactly. **Adding BM25 costs nothing on semantic recall.** You get the exact-token win for free.

The end-to-end `scripts/demo.py` shows why this compounds in a real plan. The incident query *"503 after deploy, timeouts under load"* (filtered to `status=resolved`, `top_k=8`) retrieves **7 exact-503 tickets via BM25 plus 1 semantically-related DB-pool-exhaustion ticket via vector** — the connection a keyword-only system would have missed and a vector-only system would have blurred. The plan then classifies root cause (all matched ground truth), summarizes the fix, joins account facts, and ranks by plan tier then recency (scale-tier accounts at $12.4k / $7.2k MRR promoted above launch-tier). Hybrid retrieval is what makes that whole chain land on the right evidence.

---

## The cost story (honest version)

turbopuffer's architecture is the reason it is cheap: it keeps vectors on **object storage** (S3/GCS) instead of holding the entire index in RAM the way classic vector databases do. RAM is the dominant cost line in a vector DB at scale; moving the resting state to object storage is where the **roughly 10x** cost advantage comes from. Warm reads still land around ~14ms and cold reads in the low hundreds of ms — comfortably inside a single interactive PromptQL turn. You pay for what you query, not for a fleet of always-on memory-resident replicas.

Now the honest part, because credibility matters more than spin:

**turbopuffer has no free tier.** Self-serve signup is open (email + plan selection at <https://turbopuffer.com/join> — no waitlist, no approval), but the lowest plan is **Launch at a $64/month minimum** (you pay the greater of usage or $64); Scale is $256/month. There is a 30-day cancel-for-refund window. A few-thousand-vector demo is technically far under the minimum, but you still commit to $64/mo just to get a key.

That pricing is not a knock on turbopuffer — **it is priced for production scale, where the 10x RAM savings dwarf a $64 floor.** But it does mean a *demo* should not require it. So this repo makes the deliberate choice:

> **The demo defaults to free, local LanceDB. turbopuffer is a one-env-var upgrade.**

```bash
# Free, zero-key, the clone-and-run default:
python scripts/demo.py

# The exact same plan, on real turbopuffer:
export TURBOPUFFER_API_KEY=...
export TURBOPUFFER_REGION=gcp-us-central1
RETRIEVAL_BRIDGE_BACKEND=turbopuffer python scripts/demo.py
```

The local default is the **on-ramp**; turbopuffer is the production destination. Framing it that way is honest about the cost floor *and* makes the case for paying it: the floor only exists because the architecture is built for scale.

---

## The anti-lock-in story

The strongest objection to "build on turbopuffer" is lock-in. This design answers it structurally, not rhetorically: every backend implements the same one-method `VectorBackend` protocol (`retrieval_bridge/backends/base.py`) and reuses the same client-side `reciprocal_rank_fusion` (`retrieval_bridge/fusion.py`). DDN never learns *which* store answered.

| Backend | Role | Hybrid |
|---|---|---|
| **LanceDB** | Open-source, embedded, object-storage-native — the turbopuffer **analog**, and the zero-key clone-and-run default | vector + native FTS + client-side RRF |
| **turbopuffer** | The headline: serverless hybrid at scale | vector + BM25 multi-query + client-side RRF |
| **pgvector** | Lowest friction inside DDN (native Postgres connector) | vector + `tsvector` + client-side RRF |

A subtle but important detail that makes the swap *real* and not cosmetic: **turbopuffer has no server-side rank fusion**, so RRF is client-side in the bridge by design — and the local backends mirror that faithfully. "Hybrid" means the same thing on every backend. There is no behavior cliff when you upgrade from LanceDB to turbopuffer; you are running the same fusion you already tested locally, just over a store that scales.

LanceDB being the *open-source analog* (not a toy) is the load-bearing part of the no-lock-in claim: if turbopuffer ever stops being the right answer, the open-source path is a drop-in, not a rewrite.

---

## Why hire the person who built this

This artifact is small on purpose, and every choice in it is a signal:

- **I understand what PromptQL actually is** — planning + deterministic execution + reasoning primitives — and I built retrieval as a *tool inside the plan*, not as a wholesale replacement for it. I am pitching the product on its own terms.
- **I found the existing seam instead of inventing one.** The lambda connector is already the extension point; I used it as designed (named pydantic return type so the `hit.id → Accounts` relationship attaches in metadata), rather than bolting on a custom gateway.
- **I proved the claim with numbers, not adjectives.** found@5 of 1/12 vs 12/12 on exact-id lookups, 5/5 vs 5/5 on semantic — measured by a script in the repo, not asserted in a slide.
- **I was honest about the weak spot.** No free tier, $64/mo floor — stated plainly, then turned into a design decision (free local default, one-env-var upgrade) that makes the demo frictionless *and* makes the case for paying the floor.
- **I designed against lock-in from the start,** because that is the first question anyone serious will ask, and "swap an env var" beats any amount of reassurance.
- **I verified every external API against live June 2026 documentation,** not training memory — which caught real breaking drift (turbopuffer's v2 API and region-scoped hosts, the `ndc_sdk_python` connector imports, the four PromptQL primitives). The receipts are in `docs/verified-facts.md`.

I build integrations that respect the product they plug into, prove their value with measurements, stay honest about trade-offs, and avoid trapping the customer. That is the person on the other end of this repo.

---

## Where to go next

- **[README](../README.md)** — the two-minute, zero-key quickstart.
- **[docs/query-plan-walkthrough.md](query-plan-walkthrough.md)** — the demo prompt and the annotated query plan PromptQL is expected to generate.
- **[docs/turbopuffer-runbook.md](turbopuffer-runbook.md)** — sign up, get a key, seed a real namespace.
- **[docs/ddn-runbook.md](ddn-runbook.md)** — register the `search_documents` command and the Postgres relationship in your DDN project.
- **[ddn/connector/search/functions.py](../ddn/connector/search/functions.py)** and **[retrieval_bridge/search.py](../retrieval_bridge/search.py)** — the deploy-side and local twins of the *same* command.
