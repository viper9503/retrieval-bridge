"""Generate a synthetic support-ticket corpus.

Support tickets are PromptQL's own benchmark domain, so the demo resonates. The
corpus is engineered to exercise *hybrid* retrieval specifically: every ticket
has both

  * semantic content  -> rewards vector search ("service is down", "users can't
    log in", "queries are slow after idle"), and
  * exact tokens       -> rewards BM25 ("HTTP 503", "ERR_DIM_384", "TCK-10042",
    "Enterprise", "429"),

so a pure-vector store misses the exact-token queries that hybrid nails. The
output is deterministic (seeded) for reproducible clone-and-run demos.

Run:  python data/generate_corpus.py   ->   data/tickets.jsonl
"""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 42
N_TICKETS = 160
OUT = Path(__file__).parent / "tickets.jsonl"

# Accounts (the structured "facts" each ticket joins to). plan_tier here is the
# authoritative account tier; it is also denormalized onto the ticket as a
# filterable attribute.
PROJECTS = [
    # id,         name,              plan_tier,    monthly_revenue, region,        seats
    ("proj-atlas",  "Atlas Analytics", "enterprise", 48000.0, "us-east-1",      320),
    ("proj-cygnus", "Cygnus Bank",     "enterprise", 88000.0, "us-east-1",      900),
    ("proj-vega",   "Vega Health",     "enterprise", 61000.0, "us-west-2",      540),
    ("proj-orion",  "Orion Logistics", "scale",       9500.0, "eu-west-1",       85),
    ("proj-pavo",   "Pavo Retail",     "scale",      12400.0, "eu-west-1",      140),
    ("proj-lyra",   "Lyra Media",      "scale",       7200.0, "eu-central-1",    60),
    ("proj-nova",   "Nova Robotics",   "launch",      1100.0, "ap-southeast-1",  12),
    ("proj-draco",  "Draco Games",     "launch",       640.0, "ap-northeast-3",   8),
]

PLAN_RANK = {"enterprise": 3, "scale": 2, "launch": 1, "free": 0}

# Each theme yields naturally-worded tickets that *imply* a root cause (we keep
# the canonical label in metadata for evaluation, but do not just paste it into
# the body) and carry distinctive exact tokens.
THEMES = [
    {
        "key": "outage_503",
        "root_cause": "upstream_overload",
        "component": "api-gateway",
        "codes": ["HTTP 503", "ERR_GW_503"],
        "subjects": [
            "Production API returning 503 errors under load",
            "Intermittent 503 Service Unavailable on search endpoint",
            "Gateway throwing 503s after traffic spike",
        ],
        "bodies": [
            "Starting around {time}, our production search API began returning {code} "
            "Service Unavailable for roughly {pct}% of requests in {region}. Latency "
            "climbed and the gateway shed load. Traffic was ~{rps} req/s, well above our "
            "usual baseline. Customers reported failed checkouts. Related: {ref}.",
            "We are seeing {code} responses spike whenever marketing sends a campaign. "
            "The upstream search service appears saturated and the gateway returns 503 "
            "rather than queueing. Region {region}, peak ~{rps} rps.",
        ],
        "resolutions": [
            "Raised upstream connection limits and enabled request queueing with a "
            "concurrency cap; added autoscaling on the search tier. 503 rate returned to 0.",
            "Scaled out the search service horizontally and tuned the gateway timeout; "
            "the 503s cleared once headroom was added.",
        ],
    },
    {
        "key": "cold_start_latency",
        "root_cause": "cold_cache",
        "component": "search-service",
        "codes": ["ERR_TIMEOUT_504", "HTTP 504"],
        "subjects": [
            "First query after idle period is very slow",
            "p90 latency spikes to hundreds of ms on cold namespaces",
            "Search timing out on the first request each morning",
        ],
        "bodies": [
            "The first query against a namespace that hasn't been touched in a while "
            "takes {ms} ms (p90), occasionally hitting {code}. Once it's warm, the same "
            "query is ~{ms2} ms. This cold-start penalty hurts our {region} morning peak. {ref}",
            "After periods of inactivity, the initial search reads from object storage and "
            "is dramatically slower (~{ms} ms) before the cache warms. Subsequent queries "
            "are fast. We need predictable latency for {region} users.",
        ],
        "resolutions": [
            "Added a scheduled warm-up job that pre-queries hot namespaces before peak "
            "traffic; p90 cold latency dropped from {ms} ms to under {ms2} ms.",
            "Pinned the high-traffic namespace to keep it cache-resident; cold-start "
            "timeouts disappeared.",
        ],
    },
    {
        "key": "rate_limit_429",
        "root_cause": "rate_limited",
        "component": "ingest",
        "codes": ["429", "ERR_RATE_429"],
        "subjects": [
            "Bulk ingest hitting 429 Too Many Requests",
            "Getting throttled with 429 during nightly sync",
            "429 errors when backfilling documents",
        ],
        "bodies": [
            "Our nightly backfill is being throttled with {code} Too Many Requests after "
            "~{rps} writes/s. The job retries but falls behind. Region {region}. {ref}",
            "We hit {code} during bulk upload of embeddings. The client doesn't back off "
            "and the batch fails. We need guidance on batch sizing and retry/backoff.",
        ],
        "resolutions": [
            "Implemented exponential backoff with jitter and larger batched writes; "
            "throughput stabilized and 429s stopped.",
            "Switched to batched upserts (fewer, larger requests) to stay under the rate "
            "limit; the nightly sync now completes on time.",
        ],
    },
    {
        "key": "auth_token_expiry",
        "root_cause": "expired_credential",
        "component": "auth",
        "codes": ["ERR_AUTH_419", "HTTP 401"],
        "subjects": [
            "SSO users logged out unexpectedly",
            "API calls failing with 401 after token refresh",
            "ERR_AUTH_419 session expired errors for Enterprise SSO",
        ],
        "bodies": [
            "Enterprise SSO users in {region} are being logged out mid-session with "
            "{code}. It looks like the refresh token isn't being honored. Started {time}. {ref}",
            "Service-to-service calls began failing with {code} after we rotated keys. The "
            "old token wasn't fully drained before cutover. Affects automated jobs.",
        ],
        "resolutions": [
            "Fixed the token refresh race and extended the grace window during key "
            "rotation; 401s cleared.",
            "Re-issued SSO certificates and corrected the clock skew on the auth node; "
            "session expiry errors stopped.",
        ],
    },
    {
        "key": "embedding_dim_mismatch",
        "root_cause": "dimension_mismatch",
        "component": "indexer",
        "codes": ["ERR_DIM_384", "ERR_DIM_1536"],
        "subjects": [
            "Vector dimension mismatch on upsert",
            "Indexing fails with ERR_DIM_384",
            "Query vector rejected: dimension does not match namespace",
        ],
        "bodies": [
            "Upserts are failing with {code}: the namespace was created with one embedding "
            "model and we switched models, so the vector dimension no longer matches. {ref}",
            "We get {code} at query time after changing embedding providers. The namespace "
            "dimension is fixed and our new query vectors are a different size.",
        ],
        "resolutions": [
            "Created a fresh namespace for the new embedding model and re-indexed; kept "
            "write-time and query-time models identical thereafter.",
            "Standardized on a single embedding model per namespace and added a CI check "
            "asserting the vector dimension before deploy.",
        ],
    },
    {
        "key": "data_sync_lag",
        "root_cause": "replication_lag",
        "component": "replication",
        "codes": ["ERR_LAG", "WARN_STALE"],
        "subjects": [
            "Search results missing recently added documents",
            "Stale reads: new records not appearing for minutes",
            "Replica lag causing inconsistent search results",
        ],
        "bodies": [
            "Documents written in {region} don't show up in search for several minutes. "
            "It looks like eventually-consistent reads are lagging the writes. {ref}",
            "Users add a record and immediately search for it but get nothing for ~{ms} ms. "
            "We need read-your-writes for this flow.",
        ],
        "resolutions": [
            "Switched the read-after-write path to strong consistency; stale reads resolved "
            "while keeping relaxed consistency for analytics.",
            "Added a short post-write consistency wait on the critical flow; users now see "
            "their records immediately.",
        ],
    },
    {
        "key": "memory_oom",
        "root_cause": "out_of_memory",
        "component": "worker",
        "codes": ["ERR_OOM_137", "OOMKilled"],
        "subjects": [
            "Worker pods OOMKilled during large batch",
            "ERR_OOM_137 crashes on big embedding jobs",
            "Out-of-memory when processing large documents",
        ],
        "bodies": [
            "Our embedding workers are crashing with {code} when a batch contains very "
            "large documents. The pod gets OOMKilled and the job restarts in a loop. {ref}",
            "Memory usage spikes and the process is killed ({code}) on documents over a few "
            "MB. We suspect we're loading everything into memory at once.",
        ],
        "resolutions": [
            "Streamed and chunked large documents instead of loading them whole; raised the "
            "memory limit modestly. OOM kills stopped.",
            "Added per-document size guards and chunking; the batch now completes without "
            "ERR_OOM_137.",
        ],
    },
    {
        "key": "tls_cert_expiry",
        "root_cause": "expired_certificate",
        "component": "edge",
        "codes": ["ERR_TLS_526", "CERT_EXPIRED"],
        "subjects": [
            "TLS certificate expired, clients can't connect",
            "ERR_TLS_526 on all HTTPS requests",
            "Certificate expiry caused a hard outage",
        ],
        "bodies": [
            "All clients in {region} suddenly fail with {code}; it turned out our TLS "
            "certificate expired and auto-renewal didn't fire. Full outage from {time}. {ref}",
            "HTTPS handshakes are failing with {code}. The cert chain expired and the renew "
            "hook silently errored last month.",
        ],
        "resolutions": [
            "Renewed the certificate and fixed the auto-renew hook; added an expiry alert 30 "
            "days out. Connectivity restored.",
            "Rotated certs and moved renewal to a monitored job with alerting; no more "
            "surprise expiries.",
        ],
    },
    {
        "key": "disk_full",
        "root_cause": "disk_full",
        "component": "storage",
        "codes": ["ERR_DISK_28", "ENOSPC"],
        "subjects": [
            "Writes failing with ENOSPC, disk full",
            "ERR_DISK_28 no space left on device",
            "Ingestion halted: out of disk",
        ],
        "bodies": [
            "Writes started failing with {code} (no space left on device) in {region}. Log "
            "rotation wasn't running and the volume filled up. {ref}",
            "We hit {code} during indexing; an unbounded cache directory consumed the disk.",
        ],
        "resolutions": [
            "Cleared the runaway cache, enabled log rotation, and expanded the volume with "
            "an 80% usage alert. Writes recovered.",
            "Added disk-usage monitoring and capped the cache size; ENOSPC errors stopped.",
        ],
    },
    {
        "key": "connection_pool",
        "root_cause": "pool_exhausted",
        "component": "db-proxy",
        "codes": ["ERR_POOL_53", "HTTP 500"],
        "subjects": [
            "Connection pool exhausted under load",
            "ERR_POOL_53: cannot acquire connection",
            "500 errors traced to exhausted DB pool",
        ],
        "bodies": [
            "Under load in {region} we exhaust the connection pool and requests fail with "
            "{code}, surfacing as {alt} to users. Started after the {time} release. {ref}",
            "Long-running queries hold connections and the pool runs dry ({code}); new "
            "requests queue and then error out.",
        ],
        "resolutions": [
            "Increased pool size, added connection timeouts, and fixed a query that held "
            "connections open; 500s cleared.",
            "Introduced a statement timeout and pool-acquire backoff; ERR_POOL_53 no longer "
            "occurs at peak.",
        ],
    },
]

QUESTION_NOISE = [
    {
        "key": "howto_filter",
        "root_cause": "question_not_incident",
        "component": "docs",
        "codes": ["N/A"],
        "subjects": [
            "How do I filter search results by plan tier?",
            "Question: combining metadata filters with vector search",
            "Best way to restrict results to a project_id",
        ],
        "bodies": [
            "Not an incident — a question. How do we filter hybrid search by an attribute "
            "like plan_tier or project_id in {region}? We want Enterprise-only results. {ref}",
            "We'd like to combine a metadata filter with semantic search. What's the "
            "recommended pattern for filtering by status and created_at?",
        ],
        "resolutions": [
            "Shared docs on attribute filtering; filters are applied as a predicate "
            "alongside the vector/BM25 query. Marked resolved.",
            "Pointed the customer to the filtering guide; they confirmed it works.",
        ],
    },
]

ALL_THEMES = THEMES + QUESTION_NOISE


def _fill(template: str, rng: random.Random) -> str:
    return template.format(
        time=rng.choice(["09:14 UTC", "last Tuesday", "02:30 UTC", "the 14:00 deploy", "midnight"]),
        region=rng.choice([p[4] for p in PROJECTS]),
        pct=rng.choice([3, 7, 12, 18, 25, 40]),
        rps=rng.choice([800, 1500, 3200, 6000, 9000]),
        ms=rng.choice([420, 680, 870, 1100, 1500]),
        ms2=rng.choice([8, 11, 14, 20, 35]),
        code="{code}",  # filled later with the chosen code
        alt=rng.choice(["HTTP 500", "blank pages", "spinner forever"]),
        ref=f"TCK-{rng.randint(10000, 10000 + N_TICKETS - 1)}",
    )


def generate() -> list[dict]:
    rng = random.Random(SEED)
    start_day = date(2024, 12, 1)
    tickets: list[dict] = []

    for i in range(N_TICKETS):
        theme = rng.choices(ALL_THEMES, weights=[6] * len(THEMES) + [2] * len(QUESTION_NOISE))[0]
        project = rng.choice(PROJECTS)
        proj_id, proj_name, plan_tier, revenue, region, seats = project

        ticket_id = f"TCK-{10000 + i}"
        code = rng.choice(theme["codes"])
        subject = rng.choice(theme["subjects"])
        body = _fill(rng.choice(theme["bodies"]), rng).replace("{code}", code)

        # ~70% resolved/closed (so the demo has resolved tickets to learn from).
        status = rng.choices(["resolved", "closed", "open"], weights=[55, 20, 25])[0]
        resolution = ""
        if status in ("resolved", "closed"):
            resolution = _fill(rng.choice(theme["resolutions"]), rng).replace("{code}", code)

        created = start_day + timedelta(days=rng.randint(0, 540))
        severity = rng.choice(["sev1", "sev2", "sev2", "sev3"])

        # `text` is what gets embedded + full-text indexed. We fold the subject,
        # body, and (when present) the resolution into one searchable document.
        text = f"[{ticket_id}] {subject}\n\n{body}"
        if resolution:
            text += f"\n\nResolution: {resolution}"

        tickets.append(
            {
                "id": ticket_id,
                "text": text,
                "subject": subject,
                "body": body,
                "resolution": resolution,
                # filterable metadata (denormalized for retrieval-time filtering)
                "project_id": proj_id,
                "plan_tier": plan_tier,
                "status": status,
                "created_at": created.isoformat(),
                "component": theme["component"],
                "error_code": code,
                "severity": severity,
                # ground-truth label for evaluating the classify stand-in
                "root_cause": theme["root_cause"],
                # structured "account" facts (the DDN Postgres-join target)
                "account": {
                    "ticket_id": ticket_id,
                    "project_id": proj_id,
                    "project_name": proj_name,
                    "plan_tier": plan_tier,
                    "monthly_revenue": revenue,
                    "account_region": region,
                    "seat_count": seats,
                },
            }
        )
    return tickets


def main() -> None:
    tickets = generate()
    with OUT.open("w") as f:
        for t in tickets:
            f.write(json.dumps(t) + "\n")
    by_status: dict[str, int] = {}
    by_tier: dict[str, int] = {}
    for t in tickets:
        by_status[t["status"]] = by_status.get(t["status"], 0) + 1
        by_tier[t["plan_tier"]] = by_tier.get(t["plan_tier"], 0) + 1
    print(f"Wrote {len(tickets)} tickets -> {OUT}")
    print(f"  by status: {by_status}")
    print(f"  by plan:   {by_tier}")


if __name__ == "__main__":
    main()
