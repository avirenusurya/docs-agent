# docs-agent

An agentic RAG assistant for developer docs. You ask a question, it rewrites it
into a search query, retrieves from a docs corpus, reranks the candidates with a
cross-encoder, and answers with citations. Every answer ships with a trace of
what the agent did at each step, and there's an eval harness that measures
whether a change to the pipeline actually helps instead of guessing.

The default corpus is the Model Context Protocol docs plus a focused slice of
the Claude developer docs. The corpus is swappable: edit `backend/corpus.yaml`,
re-run the ingest, and the assistant answers over whatever you point it at.

## How it works

A question runs through a small agent loop:

1. The model rewrites the question into a focused search query.
2. Retrieval pulls a wide set of candidate chunks (bi-encoder, vector search).
3. A cross-encoder reranks those candidates and keeps the best few.
4. The model reads the top chunks and either searches again or answers with
   `[n]` citations back to the sources.

The loop runs on a small JSON protocol rather than native function calling, so
it works on any chat model, including the free ones on OpenRouter. Each step is
recorded, which is what the trace panel in the UI shows.

## Design choices

- **Provider-agnostic.** The chat model, embedder, reranker, and vector store
  each sit behind a small interface. Local dev runs ONNX models (fastembed, no
  torch) and an in-process store, so there's nothing to spin up. The deployed
  instance points the same code at a hosted embedding/rerank API and pgvector
  on Supabase. One code path, two setups.

- **The reranker is measured, not assumed.** The eval harness compares the
  pipeline with the cross-encoder on and off. On this corpus the reranker takes
  recall@5 from 0.88 to 0.96, which is the number that matters because the agent
  answers from the top 5 chunks. It costs some top-1 precision (hit@1 0.76 to
  0.56, MRR 0.82 to 0.72), which would matter more for a "single best link"
  feature than for feeding context to an LLM. Reranking is on by default for
  that reason, and the harness is what made the call defensible.

- **The agent is observable.** Every request returns its steps: the rewritten
  query, which chunks came back, the rerank scores, and which sources the answer
  cited. No guessing about why it said what it said.

## Stack

- Backend: Python, FastAPI
- Frontend: React, Vite
- Vector store: pgvector on Supabase (in-process store for local dev)
- Models: OpenRouter for chat. ONNX embed/rerank locally, Jina API in deploy

## Run it locally

Backend:

```
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-local.txt
cp .env.example .env          # add your OpenRouter key
python -m app.rag.ingest      # build the index (downloads the corpus once)
uvicorn app.main:app --reload
```

Frontend:

```
cd frontend
npm install
cp .env.example .env          # VITE_API_BASE defaults to localhost:8000
npm run dev
```

## Eval

```
cd backend
python -m app.eval.run         # retrieval metrics, reranker on vs off
python -m app.eval.run --judge # also grade end-to-end answers (LLM-as-judge)
```

The test set lives in `backend/app/eval/testset.yaml`: questions paired with the
doc page that answers them. Retrieval metrics need no API key. The judge runs the
agent on each question and grades the answer against the docs it retrieved.

## Deploy

- **Database:** a Supabase Postgres with the `vector` extension. Put the
  connection string in `DATABASE_URL`.
- **Ingest into it once:** run the ingest locally with `VECTOR_STORE=pgvector`,
  `EMBEDDING_PROVIDER=jina`, `EMBEDDING_DIM=1024`, and the database and Jina keys
  set, so the embeddings in the index match what the deployed API will query with.
- **Backend:** Render free web service. `render.yaml` has the blueprint; set the
  secrets (`OPENROUTER_API_KEY`, `JINA_API_KEY`, `DATABASE_URL`) and `CORS_ORIGINS`
  in the dashboard.
- **Frontend:** Vercel. Root directory `frontend`, build `npm run build`, output
  `dist`, and set `VITE_API_BASE` to the Render URL.

## Layout

```
backend/
  app/
    providers/   chat, embeddings, rerank backends + the registry
    rag/         corpus loader, chunker, vector store, retrieval pipeline
    agent/       the loop, prompts, and trace types
    eval/        test set, metrics, judge, runner
  corpus.yaml    which docs to ingest
frontend/        React + Vite chat UI with the trace panel
```
