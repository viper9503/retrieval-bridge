# Recruiter blurb + demo script

Copy-paste material for LinkedIn, job applications, and a live/recorded walkthrough.

---

## One-liner

> I built a working integration that plugs **turbopuffer** (serverless hybrid vector + keyword search) into **Hasura PromptQL** as a swappable retrieval backend — through the Python lambda connector PromptQL already ships — and proved with measurements that hybrid retrieval fixes pure-vector's exact-token blind spot. Runs with zero API keys.

---

## Short blurb (LinkedIn / application paste)

> **retrieval-bridge** — turbopuffer as a swappable hybrid-retrieval backend for PromptQL.
>
> PromptQL (Hasura's agentic data platform) turns a question into a deterministic Python query plan and reasons over the results. It needs a way to narrow millions of documents to the handful worth reasoning about. I wired in turbopuffer — serverless vector **+** BM25 search on object storage — as a first-class `search_documents` command via the DDN Python lambda connector, then joined retrieved tickets to their structured Postgres facts through a declarative DDN relationship.
>
> The retrieval backend sits behind a one-method interface, so turbopuffer is the default but **not** a lock-in: LanceDB (its open-source analog) and pgvector drop in unchanged. The whole demo clones and runs with no API keys, ships a benchmark proving hybrid beats pure-vector on exact-identifier lookups (found@5 **1/12 → 12/12**) without hurting semantic recall, and every external API was verified against live docs. CI-tested, MIT-licensed.
>
> Repo: github.com/viper9503/retrieval-bridge

---

## ~60-second spoken demo script

Run `python scripts/demo.py` and `python scripts/bench.py` while narrating:

1. **(0:00) The setup.** "PromptQL plans and reasons; it needs high-recall, cheap retrieval to feed it. That's turbopuffer. I connected them through the lambda connector DDN already provides — no custom plumbing."
2. **(0:12) The plan.** "Here's a real incident — a 503 storm after a deploy. PromptQL generates a plan: call `search_documents`, then classify, summarize, join account facts, and rank."
3. **(0:25) Hybrid retrieval.** "The retrieval pulls seven exact-503 tickets via BM25 **and** one semantically-related connection-pool ticket via vectors — the link a keyword-only system misses and a vector-only system blurs."
4. **(0:38) Reason + join + rank.** "It classifies each root cause, summarizes the fix, joins each ticket to its account — plan tier and revenue — via a declarative relationship, and ranks scale-tier accounts above launch."
5. **(0:48) The proof.** "The benchmark makes the case concrete: on exact-identifier lookups, pure vector finds the right doc in the top 5 once out of twelve; hybrid, every time. On semantic queries, hybrid ties vector — you get the keyword win for free."
6. **(0:58) The close.** "And the backend is swappable — same command on LanceDB or pgvector — so there's no lock-in. Runs with zero keys."

---

## Talking points (for follow-up questions)

- **Why not naive RAG?** Retrieval here is a *deterministic tool inside the plan*, not the whole answer — which is exactly how PromptQL is designed to work. The plan still does the joining and reasoning.
- **Why hybrid?** Embeddings blur exact tokens (error codes, SKUs, IDs) into the same region of vector space; BM25 pins them. Fused with reciprocal-rank fusion, you get meaning and exactness. (turbopuffer has no server-side fusion, so RRF is client-side by design — and the local backends mirror that, so "hybrid" means the same thing everywhere.)
- **Cost?** turbopuffer keeps vectors on object storage instead of RAM — roughly 10× cheaper at scale. No free tier ($64/mo floor), which is precisely why the demo defaults to free local LanceDB and treats turbopuffer as a one-env-var upgrade.
- **Lock-in?** One `VectorBackend` interface + shared RRF; DDN never learns which store answered. LanceDB is the open-source drop-in.
- **Correctness?** Every external API verified against live June-2026 docs (caught turbopuffer's v2/region API, the `ndc_sdk_python` imports, Voyage deprecations); adversarial review caught and fixed three real bugs. See [verified-facts.md](verified-facts.md).
