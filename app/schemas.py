from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str


class Hit(BaseModel):
    chunk: Chunk
    score: float


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]


class LLMMessage(BaseModel):
    """A model turn: either final text or a list of tool calls."""

    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class Step(BaseModel):
    kind: str  # "tool_call" | "tool_result" | "final"
    detail: dict[str, Any] = Field(default_factory=dict)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class AnswerResult(BaseModel):
    """Structured output returned by the agent (validated, never free text)."""

    answer: str
    citations: list[str]
    tools_used: list[str]
    steps: list[Step]
    latency_ms: float
    llm_calls: int


class IngestRequest(BaseModel):
    doc_id: str = Field(min_length=1, max_length=100)
    text: str = Field(min_length=1)
