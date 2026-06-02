"""Hasura DDN Python lambda connector — the `search_documents` Command seam.

This is the deploy-side twin of `RetrievalBridge.search_documents`: the single
turbopuffer-style hybrid-retrieval command that PromptQL plans against. The whole
swappable-backend pitch lives right here — DDN never learns *which* vector store
answered. The connector calls the bridge; the bridge reads
`RETRIEVAL_BRIDGE_BACKEND` (turbopuffer for the headline deploy, LanceDB for the
zero-key clone-and-run, pgvector optional) and runs the identical vector-ANN +
BM25 sub-queries fused client-side with reciprocal-rank fusion. turbopuffer has
no server-side rank fusion, so RRF stays in the bridge no matter the backend, and
this Command signature never changes.

Two DDN facts shape this file:

  * The return type is a *named* pydantic object (`SearchHit`) with explicit
    scalar fields, not `dict`/`Any`. DDN infers the NDC object type from the type
    hints, and only a named type with a concrete `id` field lets the
    command-result -> Accounts (Postgres) relationship attach in the metadata.
  * The docstring below becomes the planner-facing Command description, so it is
    written for the query planner, not for humans reading code.

turbopuffer latency note for the deploy: cold reads off object storage run a few
hundred ms, warm reads ~14ms — well inside an interactive PromptQL turn.

Run locally:  python ddn/connector/search/functions.py
Deploy:       ddn connector init search -i   (then point env at a seeded namespace)
"""

from __future__ import annotations

from typing import Optional

from ndc_sdk_python import start
from ndc_sdk_python.function_connector import FunctionConnector
from pydantic import BaseModel, Field

connector = FunctionConnector()

# Built lazily on first query so the connector process starts fast and the
# heavy retrieval stack (embedder model, backend client) is imported only once,
# then reused across calls. See `_get_bridge`.
_bridge = None


def _get_bridge():
    """Lazily construct and cache the module-level RetrievalBridge.

    The bridge resolves its backend/embedder from env (`RETRIEVAL_BRIDGE_BACKEND`,
    `RETRIEVAL_BRIDGE_EMBEDDER`), so this connector is identical whether it is
    talking to turbopuffer, LanceDB, or pgvector.
    """
    global _bridge
    if _bridge is None:
        from retrieval_bridge import RetrievalBridge

        _bridge = RetrievalBridge()
    return _bridge


class SearchHit(BaseModel):
    """One retrieved support ticket, flattened to scalar fields.

    The flat shape (rather than a nested `metadata` dict) is deliberate: DDN turns
    each scalar field into a column of the Command's object type, which is what the
    planner filters/orders on and what the `id` relationship joins to the Accounts
    Postgres model.
    """

    id: str = Field(description="Stable ticket id; the join key to the Accounts model.")
    score: float = Field(description="Fused (vector + BM25) relevance score; higher is better.")
    text: str = Field(description="The ticket body text.")
    project_id: str = Field(description="Owning project id; foreign key into account/project facts.")
    plan_tier: str = Field(description="Customer plan tier, e.g. free | launch | scale | enterprise.")
    status: str = Field(description="Ticket status, e.g. open | resolved.")
    created_at: str = Field(description="ISO-8601 creation timestamp; use for recency ordering.")
    component: str = Field(description="Affected component/subsystem, e.g. search-api, db-pool.")
    error_code: str = Field(description="Exact error token if present, e.g. ERR_DIM_384.")
    severity: str = Field(description="Severity label, e.g. low | medium | high | critical.")
    root_cause: str = Field(description="Recorded root cause for resolved tickets, if known.")


@connector.register_query
def search_documents(
    query: str,
    top_k: int = 5,
    plan_tier: Optional[str] = None,
    status: Optional[str] = None,
) -> list[SearchHit]:
    """Hybrid semantic + keyword search over the support-ticket corpus.

    Runs a single retrieval that blends vector (meaning-based) similarity with
    BM25 (exact-keyword) matching, so it finds tickets by *concept* AND by literal
    tokens like error codes (ERR_DIM_384), ticket ids, and plan-tier names. Returns
    the most relevant tickets ranked best-first, each with its id, fused relevance
    score, full text, and metadata (project_id, plan_tier, status, created_at,
    component, error_code, severity, root_cause).

    Call this whenever the user wants to find, look up, or surface support tickets
    or incidents — e.g. "find similar tickets", "what past incidents look like
    this", "related/previous tickets about <symptom or error code>", or any
    question that needs grounding in the ticket history before classifying,
    summarizing, or joining account facts.

    Args:
        query: Natural-language description, symptom, or exact token to search for.
        top_k: Maximum number of tickets to return (best-first). Defaults to 5.
        plan_tier: Optional exact filter on plan tier (e.g. "enterprise").
        status: Optional exact filter on ticket status (e.g. "resolved").

    Returns:
        Up to `top_k` SearchHit rows, most relevant first.
    """
    bridge = _get_bridge()

    # Translate the two optional, planner-supplied facets into the bridge's
    # backend-agnostic equality-filter dict. Each backend renders this into its own
    # filter DSL (turbopuffer tuple DSL, LanceDB SQL, pgvector WHERE).
    filters: dict[str, str] = {}
    if plan_tier is not None:
        filters["plan_tier"] = plan_tier
    if status is not None:
        filters["status"] = status

    hits = bridge.search_documents(query, top_k=top_k, filters=filters or None)

    results: list[SearchHit] = []
    for hit in hits:
        md = hit.metadata or {}
        results.append(
            SearchHit(
                id=hit.id,
                score=hit.score,
                text=hit.text,
                project_id=str(md.get("project_id", "")),
                plan_tier=str(md.get("plan_tier", "")),
                status=str(md.get("status", "")),
                created_at=str(md.get("created_at", "")),
                component=str(md.get("component", "")),
                error_code=str(md.get("error_code", "")),
                severity=str(md.get("severity", "")),
                root_cause=str(md.get("root_cause", "")),
            )
        )
    return results


if __name__ == "__main__":
    start(connector)
