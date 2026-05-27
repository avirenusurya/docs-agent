# docs-agent

An agentic RAG assistant for developer docs. Ask a question, it retrieves from
a docs corpus, reranks the candidates with a cross-encoder, and answers with
citations. Every answer comes with a trace of what the agent did at each step,
and there's an eval harness that measures whether a change to the pipeline
actually helps instead of guessing.

The default corpus is the Model Context Protocol and Anthropic docs. The corpus
is swappable: point the ingest script at another docs source and re-run it.

## Why it's built this way

- **Provider-agnostic.** The chat model, embedder, reranker, and vector store
  are each behind a small interface. Local dev runs ONNX models and an
  in-process store with no external services. The deployed instance points the
  same code at a hosted embedding/rerank API and pgvector on Supabase.
- **The reranker earns its place.** First-pass vector search is fast but blunt.
  A cross-encoder rescoring the top candidates is more accurate, and the eval
  harness reports the difference (recall@k and MRR with the reranker on vs off).
- **The agent is observable.** Each request records its steps (query rewrite,
  retrieval, rerank scores, final answer) so you can see why it answered the
  way it did.

## Stack

- Backend: Python + FastAPI
- Frontend: React + Vite
- Vector store: pgvector on Supabase (local in-process store for dev)
- Models: OpenRouter for chat; ONNX embed/rerank locally, hosted API in deploy

## Status

In progress. See the build below.

## Layout

```
backend/    FastAPI app, retrieval, agent, eval harness
frontend/   React + Vite chat UI with the trace panel
```

## Running locally

```
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your OpenRouter key
uvicorn app.main:app --reload
```

Then `GET /health` and `GET /config` to confirm it's wired up.
