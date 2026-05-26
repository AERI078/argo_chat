# ArgoChat

A natural-language interface to the global Argo ocean float network. Ask questions about ocean temperature, salinity, and circulation in plain English. Get grounded, transparent answers backed by real float profiles and curated oceanographic knowledge.

**Live:** [argochat.streamlit.app](https://argochat.streamlit.app) &nbsp;·&nbsp; **API:** [floatchat-api.onrender.com](https://floatchat-api.onrender.com/health)

---

## What is Argo data?

The Argo program is a global array of over 4,000 autonomous robotic floats drifting through the world's oceans. Every ten days, each float descends to 2,000 metres, rises slowly toward the surface while measuring temperature, salinity, and pressure at each depth, then transmits the profile via satellite. The program has operated continuously since 2000 and produces over 100,000 vertical profiles per year — the most complete picture of the ocean interior ever assembled.

This data is freely available through the Argo ERDDAP servers. The problem is that accessing it requires knowing which server to query, which Python library to use, how to interpret dbar as a depth proxy, and what PSU actually means. ArgoChat removes that barrier.

---

## The Problem

Ocean data has a last-mile problem. The data is global, free, and scientifically invaluable — relevant to climate modelling, fisheries management, shipping logistics, academic research, and environmental journalism. But it is effectively locked behind a tooling wall that filters out everyone except specialists.

This is a distribution problem, not a data problem. And distribution problems are exactly what language models are well-positioned to solve — provided they are built with grounding constraints that prevent the model from filling data gaps with invention.

---

## Architecture

ArgoChat is a multi-agent RAG system. The core insight is that a single LLM call — retrieve context, answer question — is insufficient for scientific data. The system needs to plan, execute, validate, and recover from failure before it synthesises an answer.

```
User query
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  FastAPI  /chat                                                  │
│  Checks app.state.ready (background init must complete first)   │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  Orchestrator                                                    │
│                                                                  │
│   PlannerAgent ──► TaskPlan (ordered steps + dependencies)      │
│        │                                                         │
│        ▼                                                         │
│   for each step:                                                 │
│        ExecutorAgent ──► runs tool ──► StepResult               │
│        ValidatorAgent ──► scores result ──► ValidationResult    │
│        PlanEvaluator ──► coherent / replan / unrecoverable      │
│              │                                                   │
│        replan? ──► ReplanEngine ──► revised TaskPlan            │
│              │                                                   │
│        final_answer step ──► SynthesiserPrompt ──► answer       │
│                                                                  │
│   Returns: OrchestratorResponse + full pipeline trace           │
└─────────────────────────────────────────────────────────────────┘
                       │
                       ▼
            Streamlit frontend renders:
            answer · confidence score · chart · trace expander
```

### Agent Responsibilities

**PlannerAgent**
Receives the raw query and calls the LLM once with a structured system prompt containing the full tool schema and exact database column names. Returns a `TaskPlan` — an ordered list of `TaskStep` objects, each specifying a tool, parameters, and dependency IDs.

Strategies applied:
- In-memory plan cache keyed by md5(query) — identical queries skip the LLM entirely
- `bypass_cache=True` flag used by ReplanEngine so a broken cached plan is never reused
- Fallback plan (rag_search → final_answer) when LLM output is unparseable
- Prompt constraint: conceptual or explanatory queries use only rag_search + final_answer — no live data fetch

**ExecutorAgent**
Runs exactly one step per call. For `final_answer` steps, calls the synthesiser prompt directly. For all other steps, calls the executor prompt, parses `Action: {...}` JSON from the LLM output using brace-balancing (not regex, which breaks on nested objects), and dispatches to the appropriate tool.

**ValidatorAgent**
Scores each step result. Short-circuits without LLM calls wherever possible:

| Step type | Validation method | Score basis |
|-----------|------------------|-------------|
| `rag_search` | Rule-based | 0.85 if docs returned, 0.2 if empty |
| `final_answer` | Rule-based | min(1.0, len(answer) / 400) |
| `fetch_*` with zero rows | Rule-based | 0.1, failure_type = strategy |
| `fetch_*` with data | LLM scoring | Semantic quality assessment |

This eliminates approximately two LLM calls per typical query.

**PlanEvaluator**
Tracks cumulative failure counts by type after every step. Thresholds from config:

```
execution    → 3 failures triggers replan, 4+ triggers unrecoverable
dependency   → 2 / 3
strategy     → 1 / 2
invalidation → 1 / 2
```

**ReplanEngine**
When PlanEvaluator signals replan, builds a targeted repair prompt with the original plan, the failed step, and the failure reason. The LLM generates a revised plan working around the specific failure. After `MAX_REPLAN_ATTEMPTS`, falls back to a completely fresh plan via `planner.plan(query, bypass_cache=True)`.

---

## RAG Pipeline

Two separate FAISS indexes serve different retrieval purposes.

```
query
  │
  ├── Float index (WHAT)
  │     Summaries of real Argo profiles: float ID, date, location,
  │     avg temperature, avg salinity, max depth
  │     Built from ERDDAP fetch on cold start, stored in Supabase Storage
  │
  └── Knowledge index (WHY)
        Chunked text from curated oceanographic documents:
        thermoclines, salinity gradients, monsoon dynamics,
        mixed layer physics, Argo instrumentation
        400-char chunks with 80-char overlap, prefixed by source filename

Both indexes go through:
  1. Dynamic k selection    — comparison queries k=8, data queries k=6, conceptual k=5
  2. FAISS vector search    — FlatL2 exact search, no approximation
  3. Re-ranking             — 60% semantic rank (FAISS position) + 40% keyword overlap

RAGResult { docs: float summaries, knowledge_docs: science context }
```

The synthesiser is explicitly instructed to use only the retrieved context. If the context is insufficient, it says so — it does not fill gaps from training data.

---

## Data Pipeline

```
fetch_profiles_by_region(lat, lon, date_start, date_end)
          │
          ├── is_region_cached()? ──► YES ──► load_profiles_from_db() ──► return (fast)
          │
          └── NO ──► ERDDAP fetch via argopy (60s timeout)
                          │
                          ├── cache_profiles()        — permanent storage
                          ├── log_fetched_region()    — marks region as cached
                          └── return DataFrame
```

Every fetch is written to Supabase PostgreSQL permanently. Repeat queries for the same region return in milliseconds. Over time the database becomes the primary data source and ERDDAP is called progressively less.

Schema:

```sql
argo_profiles (
    float_id      VARCHAR,
    lat           FLOAT,
    lon           FLOAT,
    date          DATE,
    pressure_dbar FLOAT,
    temperature_c FLOAT,
    salinity_psu  FLOAT,
    UNIQUE (float_id, date, pressure_dbar)
)

fetched_regions (
    lat_min, lat_max, lon_min, lon_max,
    date_start, date_end,
    row_count, fetched_at
)
```

Indexes on lat, lon, date, float_id, and a composite (lat, lon, date) for region queries.

---

## Startup Sequence

```
Docker container starts on Render
    │
    ├── FastAPI binds port immediately
    │   /health returns 200 — Render health check passes instantly
    │
    └── Background thread:
            setup_tables()         — idempotent DDL + constraint migration
            RAGPipeline.__init__()
                try local disk     — miss on fresh container
                try Supabase Storage download  — hit → 5-10 seconds
                if miss: fetch ERDDAP + build + upload to Supabase
            KnowledgePipeline.__init__()  — same pattern
            wire agents
            app.state.ready = True
            /chat now accepts requests
```

The Supabase Storage layer means every deploy after the first one skips the 90-second ERDDAP cold start. FAISS indexes survive container restarts.

---

## Confidence Scoring

After all steps complete:

```python
confidence = (final_answer_score * 0.5) + (avg_other_steps * 0.5)
```

The final_answer score is weighted at 50% because it directly reflects synthesis quality. Intermediate step scores (retrieval quality, fetch success) contribute the other 50%. Displayed to the user as high / medium / low with a percentage.

---

## Tools

| Tool | Purpose | DB interaction |
|------|---------|----------------|
| `rag_search` | Semantic search over float summaries and knowledge docs | Read FAISS |
| `fetch_region` | Live Argo profiles for a lat/lon/date range from ERDDAP | Write + log |
| `fetch_float` | Full history for a specific float ID | Write + log |
| `db_query` | SELECT against cached profiles in Postgres | Read only |
| `generate_chart` | Converts rows into depth_profile / trajectory / time_series Plotly spec | None |
| `final_answer` | Synthesises all accumulated context into a grounded answer | None |

---

## Unit Economics

**Cost to serve one query (current)**

| Component | Cost per query |
|-----------|---------------|
| Groq API — Llama 3.3 70B, ~4-7 calls | ~$0.0006 |
| ERDDAP fetch | $0 (free, cached after first call) |
| Supabase DB + Storage | $0 (free tier) |
| Render hosting | $0 (free tier) |
| **Total marginal cost** | **< $0.001** |

**What drives cost down over time**

The DB-first fetch architecture means ERDDAP is called once per region per date window — ever. Every subsequent query for the same region costs only the LLM calls. The planner cache eliminates LLM calls for repeated identical queries. Validator short-circuits eliminate two LLM calls per query by scoring retrieval and synthesis steps with rule-based logic rather than asking the model.

**At scale**

| Scenario | Revenue/query | Cost/query | Gross margin |
|----------|--------------|-----------|--------------|
| $10/month · 500 queries | $0.020 | $0.001 | 95% |
| $49/month · 2,000 queries | $0.025 | $0.001 | 96% |
| API access · $0.05/query | $0.050 | $0.003 | 94% |

Primary cost driver at scale is LLM inference. Switching planning and validation steps to a smaller model (Llama 3.1 8B) while keeping the large model only for synthesis reduces per-query LLM cost by approximately 60%.

---

## User Demographics

**Primary — Academic researchers and graduate students**
Oceanography, climate science, marine biology, and environmental science programs. Currently the only users who can access Argo data, but spend significant time on data engineering rather than analysis. ArgoChat eliminates that overhead.

**Secondary — Climate and science journalists**
Need to report on ocean temperature anomalies, salinity shifts, or heat content changes but lack the technical skills to query ERDDAP directly. A natural-language interface opens Argo to this entire professional class.

**Tertiary — Policy analysts and NGOs**
Ocean health is increasingly relevant to climate policy, fisheries regulation, and coastal management. Analysts who understand the domain but not the tooling benefit directly.

**Emerging — Port logistics and maritime operators**
Ocean temperature and salinity stratification affects shipping fuel consumption, route planning, and equipment maintenance. The data has commercial value that is currently untapped outside academic circles.

**What unites all of them:** domain knowledge without tooling knowledge. The bottleneck is not interest in the data — it is the activation energy required to access it.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Llama 3.3 70B via Groq API |
| Orchestration | Custom multi-agent loop (Python) |
| Vector search | FAISS FlatL2 (local) |
| Embeddings | all-MiniLM-L6-v2 via fastembed |
| Ocean data | Argo ERDDAP via argopy |
| Database | Supabase PostgreSQL |
| Index storage | Supabase Storage |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Deployment | Render (API) + Streamlit Cloud (UI) |

---

## Future Scope

**Hybrid search**
Combine FAISS semantic retrieval with structured PostgreSQL filters on lat, lon, date, and float_id. A query like "high-salinity profiles in the Arabian Sea during monsoon season 2023" should use both vector similarity and SQL range filters — neither alone is sufficient. This requires a query decomposition step that separates the semantic intent from the structured constraints.

**Expanded regional coverage**
The current FAISS index covers the Arabian Sea 2023 window. A systematic prefetch pipeline — running nightly across eight named ocean regions, storing results permanently — would give the system a continuously growing, globally representative knowledge base without any user-triggered ERDDAP calls.

**Streaming responses**
The current architecture waits for the full multi-agent loop to complete before returning. Streaming partial results — showing the plan, then each step result as it completes — would dramatically improve perceived responsiveness without changing any backend logic.

**Biogeochemical variables**
Argo BGC floats measure oxygen, nitrate, pH, and chlorophyll in addition to the core T/S/P variables. Extending the data pipeline to ingest BGC profiles opens the system to questions about ocean productivity, carbon cycling, and deoxygenation — domains with significant research and policy relevance.

**User memory and session continuity**
Currently each query is stateless. A session layer that retains retrieved context across turns — so "now show me the same region in winter" is unambiguous — would make the system significantly more useful for iterative analysis workflows.

**Confidence calibration**
The current confidence score is rule-based and untested against ground truth. A calibration dataset — queries with known correct answers — would allow the scoring thresholds to be tuned empirically. This is the difference between a confidence score that is decorative and one that is genuinely informative.

**Dashboard and historical analysis**
A pre-computed statistics layer aggregating float counts, temperature trends, and salinity anomalies by region and season — updated automatically as new data is fetched — would give non-technical users a visual entry point before they engage with the chat interface.

---

## Local Setup

```bash
# clone and install
git clone https://github.com/yourname/argochat
cd argochat
pip install -r requirements.txt

# environment variables
cp .env.example .env
# fill in: GROQ_API_KEY, DATABASE_URL, SUPABASE_URL, SUPABASE_KEY

# start backend
uvicorn api.main:app --host 0.0.0.0 --port 8000

# start frontend (separate terminal)
BACKEND_URL=http://localhost:8000 streamlit run frontend/app.py
```

First startup downloads FAISS indexes from Supabase Storage (~10 seconds). If the bucket is empty it fetches from ERDDAP and builds the index (~90 seconds).

---

## Project Structure

```
argochat/
├── api/
│   ├── main.py                  FastAPI app, background init, CORS
│   └── routes/
│       ├── chat.py              POST /chat
│       └── health.py            GET /health
├── agents/
│   ├── orchestrator.py          Main multi-agent loop
│   ├── planner.py               Query → TaskPlan
│   ├── executor.py              TaskStep → StepResult
│   ├── validator.py             StepResult → ValidationResult
│   ├── plan_evaluator.py        Cumulative failure tracking
│   ├── replan_engine.py         Failure → revised TaskPlan
│   ├── contracts.py             Pydantic data models
│   ├── tools.py                 Tool implementations + schemas
│   ├── prompt.py                All LLM prompts
│   ├── llm_caller.py            Groq API wrapper
│   ├── parser.py                Action JSON extraction
│   └── logger.py                Structured pipeline logger
├── rag/
│   ├── pipeline.py              Dual-index RAG orchestration
│   ├── retriever.py             Dynamic k + re-ranking
│   ├── knowledge_pipeline.py    Knowledge document index
│   ├── vector_store.py          FAISS + Supabase Storage I/O
│   ├── embedder.py              fastembed wrapper
│   └── summarizer.py            DataFrame → float profile summaries
├── data_pipeline/
│   ├── fetcher.py               DB-first fetch with ERDDAP fallback
│   └── db.py                    Supabase PostgreSQL operations
├── knowledge/                   Curated oceanographic .txt files
├── frontend/
│   └── app.py                   Streamlit split-panel UI
├── config.py                    All constants and env vars
├── Dockerfile
└── render.yaml
```

---

## Limitations

**ERDDAP reliability.** The Argo ERDDAP server is a public research service, not a commercial API. It has rate limits, occasional downtime, and variable response times. Queries for large regions or long date ranges reliably timeout at 60 seconds. The DB-first architecture mitigates this over time but does not eliminate it.

**Index staleness.** The FAISS float summaries index is built from a fixed historical window. It does not update automatically as new Argo profiles are collected. A scheduled rebuild pipeline is needed for production use.

**Single-region knowledge base.** The current knowledge documents focus on the Indian Ocean and general Argo methodology. Questions about Pacific or Atlantic dynamics may receive less grounded answers.

**No authentication.** The current deployment has no user authentication or rate limiting. This is appropriate for a research prototype but not for a production system.

---

*Built as a capstone project for a Generative AI course. The goal was not to build a chatbot but to demonstrate that scientific datasets with real retrieval and grounding constraints are a better benchmark for AI system design than general-purpose assistants.*
