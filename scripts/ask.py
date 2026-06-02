"""Interactive retrieval explorer — type any query, see hybrid search respond.

A lightweight way to poke the corpus (great for a live demo): it runs just the
`search_documents` step (vector + BM25 + RRF) and prints the ranked hits. For the
full PromptQL-style plan (classify -> summarize -> join -> rank) use demo.py.

One-shot:
    python scripts/ask.py "users can't log in, 401 after SSO"
    python scripts/ask.py "ERR_DIM_384" --top-k 5

Interactive REPL (type queries until you hit enter on an empty line):
    python scripts/ask.py
    python scripts/ask.py --status resolved --plan-tier enterprise
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval_bridge import RetrievalBridge


def show(bridge: RetrievalBridge, query: str, top_k: int, filters: dict[str, str] | None) -> None:
    hits = bridge.search_documents(query, top_k=top_k, filters=filters)
    if not hits:
        print("  (no results — run `python scripts/seed.py` first, or loosen --status/--plan-tier)\n")
        return
    print(f"  {'#':<2} {'ticket':<10} {'score':>7}  {'error':<12} {'tier':<10} {'status':<9} subject")
    print("  " + "-" * 96)
    for i, h in enumerate(hits, 1):
        md = h.metadata
        subject = h.text.splitlines()[0].split("] ", 1)[-1]
        print(
            f"  {i:<2} {h.id:<10} {h.score:>7.4f}  {str(md.get('error_code','')):<12} "
            f"{str(md.get('plan_tier','')):<10} {str(md.get('status','')):<9} {subject[:46]}"
        )
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive hybrid-retrieval explorer.")
    ap.add_argument("query", nargs="*", help="Query text. Omit for an interactive REPL.")
    ap.add_argument("-k", "--top-k", type=int, default=5, help="Results to show (default 5).")
    ap.add_argument("--status", default=None, help="Optional status filter (e.g. resolved).")
    ap.add_argument("--plan-tier", default=None, help="Optional plan-tier filter.")
    args = ap.parse_args()

    filters: dict[str, str] = {}
    if args.status:
        filters["status"] = args.status
    if args.plan_tier:
        filters["plan_tier"] = args.plan_tier
    filters = filters or None

    bridge = RetrievalBridge()
    print(f"[backend: {bridge.backend.name}  embedder: {bridge.embedder.name}"
          + (f"  filters: {filters}]" if filters else "]"))

    if args.query:  # one-shot mode
        q = " ".join(args.query)
        print(f"\nquery: {q}\n")
        show(bridge, q, args.top_k, filters)
        return

    # interactive REPL
    print("Type a query and press enter. Empty line or Ctrl-D to quit.\n")
    while True:
        try:
            q = input("ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        print()
        show(bridge, q, args.top_k, filters)


if __name__ == "__main__":
    main()
