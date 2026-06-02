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

    python scripts/seed.py && python scripts/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval_bridge import RetrievalBridge, StructuredStore
from retrieval_bridge.demo_primitives import classify_root_cause, summarize_fix

PLAN_RANK = {"enterprise": 3, "scale": 2, "launch": 1, "free": 0}

INCIDENT = (
    "We just had a production incident: right after the 14:00 deploy, our search API "
    "started returning 503 Service Unavailable and requests timed out under heavy load. "
    "EU customers were hit hardest. Which past resolved incidents look like this, what "
    "was the root cause of each, and how were they fixed?"
)

BAR = "=" * 92
DASH = "-" * 92


def hr(title: str) -> None:
    print(f"\n{BAR}\n{title}\n{BAR}")


def main() -> None:
    bridge = RetrievalBridge()
    store = StructuredStore()

    hr("INCOMING INCIDENT (natural language)")
    print(INCIDENT)

    hr("GENERATED QUERY PLAN  (what PromptQL produces, executed deterministically)")
    print(
        "  1. search_documents(query=<incident>, top_k=8, filters={status:'resolved'})\n"
        "       -> hybrid vector + BM25 retrieval over the ticket corpus  [the retrieval bridge]\n"
        "       -> store the hits as an artifact\n"
        "  2. for each hit: classify(root_cause)        [PromptQL executor.classify]\n"
        "  3. for each hit: summarize(the fix)          [PromptQL executor.summarize]\n"
        "  4. for each hit: join account facts by id    [DDN relationship -> Postgres model]\n"
        "  5. rank by plan tier, then recency           [deterministic Python]"
    )

    # ---- Step 1: retrieval (the bridge) ----------------------------------
    hr("STEP 1 — search_documents()  [backend: %s, embedder: %s]" % (bridge.backend.name, bridge.embedder.name))
    hits = bridge.search_documents(INCIDENT, top_k=8, filters={"status": "resolved"})
    if not hits:
        print("No results. Did you run `python scripts/seed.py` first?")
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


if __name__ == "__main__":
    main()
