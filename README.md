# agentic-rag-evals

A small, production-minded **tool-calling RAG agent** with a **regression-gating evaluation harness**.
FastAPI service, hybrid retrieval (BM25 + embeddings, merged with Reciprocal Rank Fusion), validated tool
calls, step limits, grounded citations, SSE streaming, and CI that fails when answer quality drops.

It runs fully offline by default (a deterministic `MockLLM`), so tests and CI cost nothing. Set
`OPENAI_API_KEY` to use a real model through the same interface.

```
 question ─▶ Agent loop ─▶ LLM (OpenAI-compatible, or MockLLM)
                │               │ tool calls (JSON schema)
                │               ▼
                │        ToolRegistry ── validates args with Pydantic
                │          ├─ search_docs ─▶ HybridIndex (BM25 + hashed embeddings, RRF)
                │          └─ calculator  ─▶ AST-based safe evaluator (no eval())
                ▼
        AnswerResult (answer, citations, tools_used, steps, latency_ms, llm_calls)
```

## Why it exists

Most RAG demos stop at "it answered". This one is built around the questions that matter in production:

| Concern | What the code does |
|---|---|
| Hallucinated sources | Citations are filtered to documents the tools actually returned (`tests/test_agent.py`) |
| Runaway agents | Hard `max_steps` cap; the loop returns a clear "step limit" answer instead of spinning |
| Bad tool arguments | Arguments validated with Pydantic; errors are returned to the model so it can recover, never raised |
| Unsafe tools | Calculator is an AST walker that rejects names, calls and attribute access |
| Unanswerable questions | Retrieval returns nothing for unrelated queries, and the agent abstains |
| Silent quality regressions | `evals/run_evals.py` scores a golden set and exits non-zero below a threshold (wired into CI) |
| Observability | Every run returns a structured step trace, latency and LLM call count |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

pytest -q                          # 27 tests
python -m evals.run_evals          # golden-set evaluation
uvicorn app.main:app --reload      # http://127.0.0.1:8000/docs
```

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "How long do customers have to request a full refund?"}'
```

Stream the agent's steps as server-sent events:

```bash
curl -N localhost:8000/ask/stream -H 'content-type: application/json' \
  -d '{"question": "How is data encrypted at rest?"}'
# event: tool_call / tool_result / final
```

Use a real model (any OpenAI-compatible endpoint):

```bash
export OPENAI_API_KEY=...        # optional: OPENAI_MODEL, OPENAI_BASE_URL
python -m evals.run_evals --live
```

Docker: `docker build -t agentic-rag-evals . && docker run -p 8000:8000 agentic-rag-evals`

## Evaluation

`evals/golden.jsonl` holds 11 cases (9 factual, 1 arithmetic, 1 unanswerable). Per case the harness records:

- **retrieval_hit / MRR**: did the expected document appear, and how high
- **answer_match**: expected facts present in the final answer
- **grounded**: every citation points to a retrieved document
- **tool_accuracy**: the expected tools were called

Current baseline with the offline `MockLLM` (`python -m evals.run_evals`):

| Metric | Result |
|---|---|
| Pass rate | 10 / 11 (0.91) |
| Retrieval hit rate | 1.00 |
| MRR | 1.00 |
| Grounded citations | 1.00 |
| Tool accuracy | 1.00 |

The one failing case (`free-limit`) is a known limitation of the mock's extractive heuristic, which picks the
wrong sentence from the correct document. Retrieval is right; answer selection is not. It is left in on purpose:
it shows the harness separating retrieval quality from answer quality, and it is the case to beat with a real
model via `--live`. CI gates on `--threshold 0.9`.

## Project layout

```
app/        agent loop, tools, retrieval, LLM clients, FastAPI app, schemas
evals/      golden set + harness
data/docs/  sample knowledge base (swap for your own .md files)
tests/      retrieval, tools, agent behaviour, API
.github/    CI: ruff, pytest, evals as a regression gate
```

## Design notes and limitations

- **Hashed embeddings** keep the project dependency-light and offline. Replace `hash_embed` with a real embedding
  model for production-quality semantic recall; the hybrid and RRF logic stays the same.
- The in-memory index is rebuilt on start and is not persistent; a vector database would replace it at scale.
- `MockLLM` is a policy stub for tests, not a language model. Real-model results will differ.
- The golden set is small. Treat it as a template for a larger, domain-specific set.

## License

MIT
