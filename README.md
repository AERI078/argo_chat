# 🌊 FloatChat — Talk to the Ocean

> *Ask questions about the world's oceans in plain English. Get answers backed by real scientific data.*

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://your-app.streamlit.app)
[![API](https://img.shields.io/badge/API-Render-46E3B7?style=for-the-badge&logo=render)](https://your-api.onrender.com/docs)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python)](https://python.org)

---

## What is FloatChat?

The world's oceans are monitored by **4,000+ autonomous Argo floats** drifting through every major ocean basin, measuring temperature, salinity, and pressure at hundreds of depths. This data is freely available — but locked behind complex file formats, specialist tools, and domain expertise that most people don't have.

**FloatChat breaks that barrier.**

Type a question in plain English. Get a real answer, backed by real Argo float data from the Indian Ocean, with charts you can actually read.

```
"Show me salinity profiles near the equator in March 2023"
"What is the temperature at 100m depth in the Arabian Sea?"
"Compare BGC parameters over the last 6 months"
"Which Argo floats are closest to the Malabar coast?"
```

No data science background needed. No SQL. No NetCDF. Just questions.

---

## Who is this for?

| User | How they use FloatChat |
|------|------------------------|
| **Oceanographic researchers** | Fast natural language querying instead of writing fetch scripts |
| **Climate policy analysts** | Accessible ocean summaries without needing domain tools |
| **Students & educators** | Explore real scientific data as part of coursework |
| **Science journalists** | Query ocean conditions for reporting without a data team |
| **General public** | Understand what's happening in the world's oceans, plainly |

---

## This is a live, deployed product

FloatChat is not a prototype or a notebook demo. It is a fully deployed, end-to-end AI system:

- **Backend** — FastAPI multi-agent system running on Render (Docker container)
- **Frontend** — Streamlit chat interface on Streamlit Cloud
- **Data** — Live Argo float data fetched on demand via argopy from the Ifremer ERDDAP server
- **AI** — Llama 3.1 70B via Groq, with RAG over a FAISS vector index of float summaries
- **Database** — PostgreSQL on Supabase caching fetched profiles for fast follow-up queries

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User (Browser)                           │
└─────────────────────┬───────────────────────────────────────┘
                      │ types a question
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Streamlit Frontend (Streamlit Cloud)           │
│   Chat UI · Plotly Charts · Agent Trace Viewer              │
└─────────────────────┬───────────────────────────────────────┘
                      │ POST /chat
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                FastAPI Backend (Render)                     │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                    Orchestrator                      │   │
│  │                                                      │   │
│  │  ┌─────────────┐   produces    ┌──────────────────┐  │   │
│  │  │PlannerAgent │──────────────▶│    Task Plan     │  │   │
│  │  └─────────────┘               └────────┬─────────┘  │   │
│  │                                         │ for each   │   │
│  │                                         │ step       │   │
│  │  ┌──────────────────────────────────────▼──────────┐ │   │
│  │  │              ExecutorAgent                      │ │   │
│  │  │  rag_search · fetch_region · db_query · chart   │ │   │
│  │  └──────────────────────────────────────┬──────────┘ │   │
│  │                                         │            │   │
│  │  ┌──────────────────────────────────────▼──────────┐ │   │
│  │  │             ValidatorAgent                      │ │   │
│  │  │    scores result · classifies failure type      │ │   │
│  │  └──────────────────────────────────────┬──────────┘ │   │
│  │                                         │            │   │
│  │  ┌──────────────────────────────────────▼──────────┐ │   │
│  │  │             PlanEvaluator                       │ │   │
│  │  │  coherent? → continue  replan? → ReplanEngine   │ │   │
│  │  └─────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌──────────────┐  ┌─────────────────┐  ┌───────────────┐  │
│  │  FAISS Index │  │ argopy / ERDDAP │  │   Supabase    │  │
│  │  (semantic)  │  │  (live data)    │  │  (SQL cache)  │  │
│  └──────────────┘  └─────────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### How a query flows through the system

1. User types a question in the Streamlit chat interface
2. Frontend sends `POST /chat` to the FastAPI backend
3. **PlannerAgent** decomposes the query into ordered steps (e.g. "search context → fetch region data → generate chart → answer")
4. **ExecutorAgent** runs each step using the right tool:
   - `rag_search` → FAISS semantic search over float summaries
   - `fetch_region` / `fetch_float` → live Argo data via argopy
   - `db_query` → SQL against cached profiles in Supabase
   - `generate_chart` → Plotly chart spec for the frontend
5. **ValidatorAgent** scores each result and classifies any failures
6. **PlanEvaluator** checks if the plan is still valid after each step
7. **ReplanEngine** generates a revised plan if needed
8. Final answer + chart spec returned to frontend
9. Streamlit renders the answer, chart, and a "How I answered this" trace panel

### Why two databases?

| | FAISS | PostgreSQL (Supabase) |
|--|--|--|
| **What it stores** | Embeddings of float profile summaries | Raw profile measurements |
| **Query type** | Semantic — "what's in the Arabian Sea" | Structured — "lat 10-20, March 2023" |
| **When it's used** | Every query (context retrieval) | After live data is fetched and cached |
| **Built** | Once at startup, saved to disk | Incrementally as users query |

---

## Directory Structure

```
floatchat/
│
├── config.py                     ← all constants and env vars
├── pipeline_logger.py            ← structured pipeline event logger
├── requirements.txt
├── Dockerfile                    ← for Render deployment
├── render.yaml                   ← Render service config
├── .env.example                  ← template for secrets
├── .streamlit/config.toml        ← Streamlit theme config
│
├── data_pipeline/
│   ├── fetcher.py                ← argopy: fetch Argo profiles on demand
│   └── db.py                     ← Supabase: setup, cache, query
│
├── rag/
│   ├── embedder.py               ← sentence-transformers embedder
│   ├── vector_store.py           ← FAISS index: build, save, load, search
│   ├── summarizer.py             ← DataFrame rows → text summaries
│   ├── retriever.py              ← query → top-k similar summaries
│   └── pipeline.py               ← RAGPipeline: loads/builds index on init
│
├── agents/
│   ├── contracts.py              ← Pydantic typed contracts for all handoffs
│   ├── llm_caller.py             ← Groq API wrapper
│   ├── parser.py                 ← extracts Action JSON from LLM output
│   ├── prompt.py                 ← per-agent system prompts
│   ├── tools.py                  ← tool functions + TOOL_SCHEMAS registry
│   ├── planner.py                ← query → TaskPlan
│   ├── executor.py               ← TaskStep → StepResult
│   ├── validator.py              ← StepResult → ValidationResult
│   ├── plan_evaluator.py         ← global plan health after each step
│   ├── replan_engine.py          ← generates revised plan on failure
│   ├── orchestrator.py           ← coordinates full multi-agent loop
│   └── factory.py                ← build_orchestrator() composition root
│
├── api/
│   ├── main.py                   ← FastAPI app + lifespan startup
│   └── routes/
│       ├── chat.py               ← POST /chat
│       └── health.py             ← GET /health
│
├── frontend/
│   └── app.py                    ← Streamlit chat UI
│
└── test_pipeline.py              ← 28 tests across 10 sections
```

---

## Setup & Running Locally

### Prerequisites
- Python 3.11+
- A [Groq API key](https://console.groq.com) (free)
- A [Supabase](https://supabase.com) project (free) — for the PostgreSQL cache

### 1. Clone and install

```bash
git clone https://github.com/yourusername/floatchat
cd floatchat
python -m venv argoenv
source argoenv/bin/activate  # Windows: argoenv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Fill in your `.env`:

```
GROQ_API_KEY=gsk_...
DATABASE_URL=postgresql://postgres.YOURREF:PASSWORD@aws-0-region.pooler.supabase.com:6543/postgres
BACKEND_URL=http://localhost:8000
```

### 3. Run the backend

```bash
uvicorn api.main:app --reload
```

On first run, FloatChat will:
1. Download the embedding model (~90MB, cached after)
2. Fetch a sample of Arabian Sea Argo profiles via argopy (~30-60s)
3. Build and save the FAISS index to disk
4. Print `Ready.` — startup complete

Every subsequent restart loads the saved index in seconds.

Visit `http://localhost:8000/docs` for the interactive API explorer.

### 4. Run the frontend

In a second terminal:

```bash
streamlit run frontend/app.py
```

Open `http://localhost:8501`

### 5. Run tests

```bash
python test_pipeline.py           # summary output
python test_pipeline.py --verbose # full tracebacks on failures
```

---

## Deploying to Production

### Backend → Render

1. Push your repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo — Render detects the `Dockerfile` automatically
4. Add environment variables in the Render dashboard:
   - `GROQ_API_KEY`
   - `DATABASE_URL`
5. Click **Deploy**
6. Copy your service URL (e.g. `https://floatchat-api.onrender.com`)

> **Note:** Free tier services sleep after 15 minutes of inactivity. First request after sleep takes ~30s to wake up.

### Frontend → Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repo
3. Set **Main file path** to `frontend/app.py`
4. Under **Secrets**, add:
   ```
   BACKEND_URL = "https://floatchat-api.onrender.com"
   ```
5. Click **Deploy**

---

## Example Queries

**For general users:**
- *"What is an Argo float and what does it measure?"*
- *"Is the Arabian Sea getting warmer?"*
- *"Show me where ocean floats are in the Indian Ocean"*

**For researchers:**
- *"Show salinity profiles near lat 15, lon 65 between June and December 2023"*
- *"Compare temperature at 100 dbar across different float IDs"*
- *"Fetch BGC data for the Bay of Bengal in Q1 2023 and plot a time series"*

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| LLM | Llama 3.1 70B via Groq | Free tier, fast LPU inference, strong SQL generation |
| Embeddings | all-MiniLM-L6-v2 | Runs locally, zero cost, good semantic quality |
| Vector store | FAISS (FlatL2) | Exact search, no external service, right-sized for dataset |
| Relational DB | PostgreSQL via Supabase | Concurrent connections, free hosted tier, geospatial ready |
| Backend | FastAPI + Uvicorn | Async I/O for concurrent LLM + DB calls |
| Frontend | Streamlit | Chat UI built-in, Plotly native, fast to ship |
| Data source | argopy → Argo ERDDAP | Official Argo data API, no auth required |
| Agent framework | Manual ReAct loop | Full transparency, debuggable, no framework lock-in |
| Deployment | Render (API) + Streamlit Cloud (UI) | Free tier, GitHub-native, zero DevOps |

---

## Limitations

**Data coverage** — The FAISS index is seeded with Arabian Sea data (2023). Queries about other regions trigger live fetches which can be slow (30-60s) on first request.

**ERDDAP timeouts** — The Argo data server occasionally times out on large regional queries. The agent handles this as an execution failure and retries, but very broad queries (e.g. entire Indian Ocean over 2 years) may fail.

**Cold starts** — Render's free tier sleeps after inactivity. First request after sleep rebuilds the connection, not the FAISS index (that's persisted to disk).

**SQL generation accuracy** — Natural language to SQL translation works well for simple filters but can fail on complex multi-join queries. The validator catches these and triggers a replan.

**Single-session memory** — The system has no cross-session memory. Each conversation starts fresh. Users cannot refer back to previous sessions.

**English only** — Query understanding is English-only. The Argo data itself is global.

---

## Future Scope

**LangSmith observability** — Wire in LangSmith tracing for production monitoring, latency dashboards, and prompt regression testing. The logger infrastructure is already in place.

**Extended data coverage** — Ingest BGC (Bio-Geo-Chemical) float data, glider observations, and satellite SST layers. The RAG pipeline is format-agnostic and can ingest any tabular source.

**LangGraph migration** — Replace the manual ReAct loop with LangGraph for durable checkpointing, interrupt/resume on long fetches, and visual workflow debugging.

**Fine-tuned SQL generator** — Fine-tune a small model (Qwen 2.5 3B via QLoRA) specifically on Argo schema SQL generation for higher accuracy and lower latency on structured queries.

**Hybrid retrieval** — Add BM25 sparse retrieval alongside FAISS dense retrieval (Reciprocal Rank Fusion) for better handling of float IDs, dates, and exact region names.

**Voice interface** — Add Whisper STT and ElevenLabs TTS for voice-driven ocean data queries — particularly useful for field researchers.

**Multi-ocean extension** — Extend beyond the Indian Ocean to global Argo coverage, with region-aware FAISS sharding for scalable semantic search.

**HITL gates** — Add human-in-the-loop confirmation for queries that would fetch very large datasets (>10,000 profiles) before executing.

---

## Acknowledgements

- **Argo Program** — the international array of profiling floats that makes this data possible ([argo.ucsd.edu](https://argo.ucsd.edu))
- **INCOIS** — Indian National Centre for Ocean Information Services, the Indian Argo data centre
- **argopy** — the open-source Python library for Argo data access ([argopy.readthedocs.io](https://argopy.readthedocs.io))
- **Problem Statement 25040** — FloatChat concept from Smart India Hackathon 2025, Ministry of Earth Sciences

---

*Built as a capstone project demonstrating production-grade multi-agent AI systems with RAG, structured data retrieval, and live deployment.*