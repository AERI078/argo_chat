# FloatChat — Project Structure

Run everything from the project root (`floatchat/`).

```
floatchat/
│
├── .env                          ← your secrets (never commit this)
├── .env.example                  ← template to copy from
├── .gitignore
├── requirements.txt
├── config.py                     ← all env vars and constants — single source of truth
├── logger.py                     ← structured pipeline logger used by all modules
│
├── data_pipeline/
│   ├── fetcher.py                ← argopy: fetch Indian Ocean profiles on demand
│   └── db.py                     ← Supabase PostgreSQL: setup, cache, query
│
├── rag/
│   ├── embedder.py               ← sentence-transformers: text → float32 vectors
│   ├── vector_store.py           ← FAISS: build, save, load, search index
│   ├── summarizer.py             ← converts DataFrame rows → text summaries for FAISS
│   ├── retriever.py              ← combines embedder + vector_store into retrieve()
│   └── pipeline.py               ← RAGPipeline: loads/builds index on init, exposes retrieve()
│
├── agents/
│   ├── contracts.py              ← Pydantic typed contracts for every agent handoff
│   ├── llm_caller.py             ← Groq API wrapper — single point for all LLM calls
│   ├── parser.py                 ← extracts Action JSON from raw LLM output
│   ├── prompt.py                 ← one system prompt per agent
│   ├── tools.py                  ← all callable tools + TOOL_SCHEMAS registry
│   │
│   ├── planner.py                ← query → TaskPlan (ordered steps)
│   ├── executor.py               ← TaskStep → StepResult (runs one tool)
│   ├── validator.py              ← StepResult → ValidationResult (scores quality)
│   ├── plan_evaluator.py         ← checks global plan health after each step
│   ├── replan_engine.py          ← generates revised plan when evaluator signals replan
│   ├── orchestrator.py           ← coordinates full loop → (OrchestratorResponse, trace)
│   └── factory.py                ← build_orchestrator() — wires everything together
│
├── api/                          ← MODULE 4 (not yet built)
│   ├── main.py                   ← FastAPI app + startup
│   └── routes/
│       ├── chat.py               ← POST /chat
│       └── health.py             ← GET /health
│
├── frontend/                     ← MODULE 5 (not yet built)
│   └── app.py                    ← Streamlit chat UI
│
├── test_pipeline.py              ← run with: python test_pipeline.py
│
├── floatchat.faiss               ← generated on first run (gitignored)
└── floatchat_docs.json           ← generated on first run (gitignored)
```

## Data flow

```
User query
  → Orchestrator.run(query)
      → PlannerAgent         produces TaskPlan
      → for each TaskStep:
          → ExecutorAgent    runs the tool, returns StepResult
          → ValidatorAgent   scores result, returns ValidationResult
          → PlanEvaluator    checks global health, returns PlanEvaluation
          → ReplanEngine     if replan signal: generates revised TaskPlan
      → returns (OrchestratorResponse, trace_dict)
  → API returns JSON to frontend
  → Streamlit renders answer + chart
```

## Key contracts between agents

```
PlannerAgent   → TaskPlan          (steps, rationale)
ExecutorAgent  → StepResult        (tool, success, data, error)
ValidatorAgent → ValidationResult  (passed, score, failure_type, reason)
PlanEvaluator  → PlanEvaluation    (status: coherent | replan | unrecoverable)
ReplanEngine   → TaskPlan          (revised steps)
Orchestrator   → OrchestratorResponse + trace dict
```

## How to run

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. copy and fill in your secrets
cp .env.example .env

# 3. run the tester (no live API calls needed for most tests)
python test_pipeline.py

# 4. run with verbose tracebacks on failure
python test_pipeline.py --verbose
```

## What the tester covers

| Section | What it tests | Live calls? |
|---------|---------------|-------------|
| 1. Config | imports, constants, env vars | No |
| 2. Logger | all log stages, trace structure | No |
| 3. Contracts | Pydantic validation | No |
| 4. Parser | valid/invalid/multiline JSON | No |
| 5. RAG components | embedder shape, FAISS search, summarizer | No (local model) |
| 6. Tools | rag_search, db_query safety, chart gen | Mocked |
| 7. Plan Evaluator | coherent/replan/unrecoverable transitions | No |
| 8. Planner | valid output, fallback, max steps cap | Mocked LLM |
| 9. Validator | execution/strategy failure detection | Mocked LLM |
| 10. Orchestrator | happy path, replan trigger, trace | Fully mocked |
