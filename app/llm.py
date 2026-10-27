"""LLM clients behind one small interface.

* OpenAIClient talks to any OpenAI-compatible /chat/completions endpoint with
  tool calling (set OPENAI_API_KEY).
* MockLLM is a deterministic, offline stand-in with a simple tool-use policy so
  tests and CI run without network access or API cost."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol

import httpx

from .retrieval import tokenize
from .schemas import LLMMessage, ToolCall

SYSTEM_PROMPT = (
    "You answer questions using ONLY information returned by tools. "
    "Call search_docs before answering factual questions and calculator for arithmetic. "
    "Cite the doc ids you used in square brackets like [doc_id]. "
    "If the tools return nothing relevant, say you don't know."
)


class LLM(Protocol):
    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMMessage: ...


class OpenAIClient:
    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, timeout: float = 30.0) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ["OPENAI_API_KEY"]
        self.client = httpx.Client(timeout=timeout)

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMMessage:
        resp = self.client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": messages, "tools": tools, "temperature": 0},
        )
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        calls = [
            ToolCall(id=c["id"], name=c["function"]["name"],
                     arguments=json.loads(c["function"]["arguments"] or "{}"))
            for c in msg.get("tool_calls") or []
        ]
        return LLMMessage(content=msg.get("content"), tool_calls=calls)


_MATH = re.compile(r"(\d[\d\s.+\-*/()%^]*\d)")


class MockLLM:
    """Policy: arithmetic question -> calculator; otherwise search_docs; after a
    tool result, answer extractively from the top passage and cite it."""

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMMessage:
        question = next(m["content"] for m in messages if m["role"] == "user")
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        if not tool_msgs:
            m = _MATH.search(question)
            if m and re.search(r"[+\-*/^%]", m.group(1)) and ("calculate" in question.lower()
                                                                 or "what is" in question.lower()):
                expr = m.group(1).replace("^", "**").strip()
                return LLMMessage(tool_calls=[ToolCall(id="c1", name="calculator",
                                                       arguments={"expression": expr})])
            return LLMMessage(tool_calls=[ToolCall(id="s1", name="search_docs",
                                                   arguments={"query": question, "k": 3})])
        payload = json.loads(tool_msgs[-1]["content"])
        if "value" in payload:
            return LLMMessage(content=f"The result is {payload['value']:g}.")
        results = payload.get("results") or []
        if not results:
            return LLMMessage(content="I don't know based on the available documents.")
        # Extractive answer: score sentences by overlap, weighting rarer terms higher.
        q_terms = set(tokenize(question))
        sents = [(r["doc_id"], x.strip()) for r in results
                 for x in re.split(r"(?<=[.!?])\s+", r["text"]) if len(x.split()) >= 4]
        df = {t: sum(t in set(tokenize(x)) for _, x in sents) or 1 for t in q_terms}
        best, best_score, best_doc, best_overlap = "", 0.0, "", 0
        for doc_id, sent in sents:
            overlap = q_terms & set(tokenize(sent))
            score = sum(1 / df[t] for t in overlap)
            if score > best_score:
                best, best_score, best_doc, best_overlap = sent, score, doc_id, len(overlap)
        if best_overlap < 2:  # weak evidence: abstain instead of guessing
            return LLMMessage(content="I don't know based on the available documents.")
        return LLMMessage(content=f"{best} [{best_doc}]")
