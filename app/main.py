"""FastAPI service: /ask (JSON), /ask/stream (SSE), /ingest, /health."""
from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from .agent import Agent
from .llm import LLM, MockLLM, OpenAIClient
from .retrieval import HybridIndex
from .schemas import AnswerResult, AskRequest, IngestRequest
from .tools import ToolRegistry

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"


def load_index(docs_dir: Path = DOCS_DIR) -> HybridIndex:
    index = HybridIndex()
    for path in sorted(docs_dir.glob("*.md")):
        index.add_document(path.stem, path.read_text(encoding="utf-8"))
    return index


def build_llm() -> LLM:
    return OpenAIClient() if os.getenv("OPENAI_API_KEY") else MockLLM()


def create_app(llm: LLM | None = None, index: HybridIndex | None = None) -> FastAPI:
    app = FastAPI(title="agentic-rag-evals", version="0.1.0")
    idx = index or load_index()
    agent = Agent(llm or build_llm(), ToolRegistry(idx))

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "chunks": len(idx.chunks), "llm": type(agent.llm).__name__}

    @app.post("/ingest")
    def ingest(req: IngestRequest) -> dict[str, object]:
        return {"doc_id": req.doc_id, "chunks_added": idx.add_document(req.doc_id, req.text)}

    @app.post("/ask", response_model=AnswerResult)
    def ask(req: AskRequest) -> AnswerResult:
        return agent.run(req.question)

    @app.post("/ask/stream")
    def ask_stream(req: AskRequest) -> StreamingResponse:
        def events() -> Iterator[str]:
            for step in agent.run_stream(req.question):
                yield f"event: {step.kind}\ndata: {json.dumps(step.detail)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


app = create_app()
