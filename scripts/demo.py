"""End-to-end demo: a natural-language incident -> a PromptQL-style query plan
executed locally over the swappable retrieval backend.

This runs the *exact division of labor* the deployed system uses, offline and
zero-key:

    1. search_documents(...)   -> turbopuffer/LanceDB hybrid retrieval  (the bridge)
    2. classify(...)           -> root-cause per hit         (PromptQL primitive*)
    3. summarize(...)          -> the fix per hit            (PromptQL primitive*)
    4. join structured facts   -> account/plan/revenue       (DDN relationship**)
    5. rank deterministically  -> by plan tier, then recency (plain Python)

    *  emulated here by retrieval_bridge.demo_primitives (see that file's note)
    ** emulated here by StructuredStore.get_by_ids (the SQLite Postgres analog)

Whichever backend you seeded (LanceDB by default, or turbopuffer) the steps and
the command signature are identical — that is the swappable-backend pitch.

Run the default (503 outage) incident:
    python scripts/demo.py

Run a built-in example, or your own free-text incident:
    python scripts/demo.py --example oom
    python scripts/demo.py --query "users can't log in, seeing 401 after SSO"
    python scripts/demo.py --list                 # show all built-in examples
    python scripts/demo.py -q "..." --top-k 5 --status resolved --plan-tier enterprise
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval_bridge import RetrievalBridge, StructuredStore
from retrieval_bridge.demo_primitives import classify_root_cause, summarize_fix

PLAN_RANK = {"enterprise": 3, "scale": 2, "launch": 1, "free": 0}

# Built-in example incidents, each phrased as a real "find past incidents like
# this" question and mapped to a theme the corpus actually covers, so retrieval
# has strong material to return. Pick one with --example <key>, list with --list,
# or pass your own with --query.
EXAMPLES: dict[str, str] = {
    "503": (
        "We just had a production incident: right after the 14:00 deploy, our search API "
        "started returning 503 Service Unavailable and requests timed out under heavy load. "
        "EU customers were hit hardest. Which past resolved incidents look like this, what "
        "was the root cause of each, and how were they fixed?"
    ),
    "auth": (
        "Enterprise SSO users are suddenly being logged out mid-session and some service "
        "calls fail with 401. What past resolved incidents match this, what was the root "
        "cause, and how was it fixed?"
    ),
    "coldstart": (
        "The very first search each morning after an idle period is extremely slow "
        "(hundreds of ms) and then speeds up once it's warm. Have we resolved anything "
        "like this before, and how?"
    ),
    "ratelimit": (
        "Our nightly bulk ingest keeps getting throttled with 429 Too Many Requests and "
        "falls behind. Find similar past tickets and how they were resolved."
    ),
    "oom": (
        "Embedding workers crash with out-of-memory (OOMKilled) on large documents and "
        "restart in a loop. Find similar resolved incidents and the fix."
    ),
    "dim": (
        "After switching embedding models, our upserts and queries fail with a vector "
        "dimension mismatch. Are there past resolved tickets like this, and what fixed them?"
    ),
    "cert": (
        "All clients in one region suddenly can't connect over HTTPS and it looks like a "
        "TLS certificate problem. What resolved incidents match this and what was the fix?"
    ),
    "disk": (
        "Writes started failing with 'no space left on device' and ingestion halted. "
        "Find similar past incidents and their resolutions."
    ),
    "pool": (
        "Under heavy load we get 500s that trace back to an exhausted database connection "
        "pool. What past resolved tickets are similar and how were they fixed?"
    ),
    "stale": (
        "Users add a record and then can't find it in search for several minutes. Find "
        "similar resolved incidents and the fix."
    ),
}

BAR = "=" * 92
DASH = "-" * 92


def hr(title: str) -> None:
    print(f"\n{BAR}\n{title}\n{BAR}")


def run_plan(query: str, top_k: int, filters: dict[str, str] | None) -> None:
    """Execute the full PromptQL-style plan for one incident query."""
    bridge = RetrievalBridge()
    store = StructuredStore()

    hr("INCOMING INCIDENT (natural language)")
    print(query)

    filt_repr = filters if filters else "none"
    hr("GENERATED QUERY PLAN  (what PromptQL produces, executed deterministically)")
    print(
        f"  1. search_documents(query=<incident>, top_k={top_k}, filters={filt_repr})\n"
        "       -> hybrid vector + BM25 retrieval over the ticket corpus  [the retrieval bridge]\n"
        "       -> store the hits as an artifact\n"
        "  2. for each hit: classify(root_cause)        [PromptQL executor.classify]\n"
        "  3. for each hit: summarize(the fix)          [PromptQL executor.summarize]\n"
        "  4. for each hit: join account facts by id    [DDN relationship -> Postgres model]\n"
        "  5. rank by plan tier, then recency           [deterministic Python]"
    )

    # ---- Step 1: retrieval (the bridge) ----------------------------------
    hr("STEP 1 — search_documents()  [backend: %s, embedder: %s]" % (bridge.backend.name, bridge.embedder.name))
    hits = bridge.search_documents(query, top_k=top_k, filters=filters)
    if not hits:
        print("No results. Did you run `python scripts/seed.py` first? "
              "(Or your filters may be too narrow — try --status any.)")
        return
    print(f"Retrieved {len(hits)} candidate tickets (fused RRF score, best first):\n")
    for h in hits:
        head = h.text.splitlines()[0]
        print(f"  {h.id}  score={h.score:.4f}  [{h.metadata.get('error_code')}]  {head[:60]}")

    # ---- Steps 2-4: primitives + structured join -------------------------
    enriched = []
    account_map = store.get_by_ids([h.id for h in hits])
    for h in hits:
        root_cause, conf = classify_root_cause(h.text)          # step 2
        fix = summarize_fix(h.text)                              # step 3
        account = account_map.get(h.id, {})                     # step 4
        enriched.append(
            {
                "id": h.id,
                "score": h.score,
                "root_cause": root_cause,
                "confidence": conf,
                "fix": fix,
                "plan_tier": account.get("plan_tier", "?"),
                "monthly_revenue": account.get("monthly_revenue", 0.0),
                "project_name": account.get("project_name", "?"),
                "created_at": h.metadata.get("created_at", ""),
                "labeled_root_cause": h.metadata.get("root_cause"),
            }
        )

    hr("STEPS 2-4 — classify root cause, summarize fix, join account facts")
    for e in enriched:
        ok = "✓" if e["root_cause"] == e["labeled_root_cause"] else "≈"
        print(
            f"  {e['id']}  classify={e['root_cause']} ({e['confidence']:.2f}) {ok}  "
            f"join-> {e['project_name']} / {e['plan_tier']} / ${e['monthly_revenue']:,.0f} MRR"
        )

    # ---- Step 5: deterministic ranking -----------------------------------
    # rank by plan tier (highest-value first), then recency (newest first).
    enriched.sort(key=lambda e: (PLAN_RANK.get(e["plan_tier"], 0), e["created_at"]), reverse=True)

    hr("STEP 5 — FINAL ANSWER  (ranked by plan tier, then recency)")
    print(f"  {'#':<2} {'ticket':<10} {'plan':<11} {'MRR':>10}  {'opened':<11} {'root cause':<20} project")
    print(f"  {DASH}")
    for i, e in enumerate(enriched, 1):
        print(
            f"  {i:<2} {e['id']:<10} {e['plan_tier']:<11} ${e['monthly_revenue']:>9,.0f}  "
            f"{e['created_at']:<11} {e['root_cause']:<20} {e['project_name']}"
        )
    print()
    for i, e in enumerate(enriched, 1):
        print(f"  {i}. {e['id']} — fix: {e['fix']}")

    hr("DIVISION OF LABOR")
    print(
        "  turbopuffer / LanceDB : scalable, cheap HYBRID RETRIEVAL (millions -> a handful)\n"
        "  PromptQL              : PLANNING + deterministic execution + classify/summarize\n"
        "  Postgres via DDN      : STRUCTURED FACTS joined to hits by a declarative relationship\n"
        "\n  Swap the backend (LanceDB <-> turbopuffer <-> pgvector): the plan above is unchanged."
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the end-to-end PromptQL-style plan on a built-in or custom incident.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("-q", "--query", help="Your own free-text incident/question.")
    ap.add_argument("-e", "--example", choices=sorted(EXAMPLES), help="Use a built-in example incident.")
    ap.add_argument("-k", "--top-k", type=int, default=8, help="How many candidates to retrieve (default 8).")
    ap.add_argument("--status", default="resolved",
                    help="Filter by ticket status (default 'resolved'; use 'any' for no status filter).")
    ap.add_argument("--plan-tier", default=None,
                    help="Optional filter by plan tier: free | launch | scale | enterprise.")
    ap.add_argument("-l", "--list", action="store_true", help="List the built-in example incidents and exit.")
    args = ap.parse_args()

    if args.list:
        print("Built-in example incidents (use with --example <key>):\n")
        for key, text in EXAMPLES.items():
            print(f"  {key:<10} {text[:78]}…")
        print("\nOr pass your own:  python scripts/demo.py --query \"<your incident>\"")
        return

    # Resolve the incident text: --query wins, else --example, else the 503 default.
    if args.query:
        query = args.query
    elif args.example:
        query = EXAMPLES[args.example]
    else:
        query = EXAMPLES["503"]

    # Build the filter dict from --status / --plan-tier.
    filters: dict[str, str] = {}
    if args.status and args.status.lower() not in ("any", "all", ""):
        filters["status"] = args.status
    if args.plan_tier:
        filters["plan_tier"] = args.plan_tier

    run_plan(query, top_k=args.top_k, filters=filters or None)


if __name__ == "__main__":
    main()
