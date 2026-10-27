"""Tool-calling agent loop with step cap, structured trace, and citation checks."""
from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from typing import Any

from .llm import LLM, SYSTEM_PROMPT
from .schemas import AnswerResult, Step
from .tools import ToolRegistry

_CITE = re.compile(r"\[([A-Za-z0-9_\-./]+)\]")


class Agent:
    def __init__(self, llm: LLM, tools: ToolRegistry, max_steps: int = 4) -> None:
        self.llm, self.tools, self.max_steps = llm, tools, max_steps
        self.llm_calls = 0

    def run_stream(self, question: str) -> Iterator[Step]:
        """Yields each step as it happens; the last step has kind='final'."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        self.llm_calls = 0
        for _ in range(self.max_steps):
            self.llm_calls += 1
            msg = self.llm.complete(messages, self.tools.schemas())
            if not msg.tool_calls:
                yield Step(kind="final", detail={"content": msg.content or ""})
                return
            messages.append({
                "role": "assistant", "content": msg.content,
                "tool_calls": [{"id": c.id, "type": "function",
                                "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                               for c in msg.tool_calls],
            })
            for call in msg.tool_calls:
                yield Step(kind="tool_call", detail={"name": call.name, "arguments": call.arguments})
                result = self.tools.execute(call.name, call.arguments)
                yield Step(kind="tool_result", detail={"name": call.name, "result": result})
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": json.dumps(result)})
        yield Step(kind="final", detail={"content": "I could not finish within the step limit.",
                                         "truncated": True})

    def run(self, question: str) -> AnswerResult:
        t0 = time.perf_counter()
        steps = list(self.run_stream(question))
        final = steps[-1].detail["content"]
        retrieved = {
            r["doc_id"]
            for s in steps if s.kind == "tool_result"
            for r in s.detail["result"].get("results", [])
        }
        # Only keep citations that point at documents the tools actually returned.
        citations = [c for c in dict.fromkeys(_CITE.findall(final)) if c in retrieved]
        return AnswerResult(
            answer=final,
            citations=citations,
            tools_used=[s.detail["name"] for s in steps if s.kind == "tool_call"],
            steps=steps,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            llm_calls=self.llm_calls,
        )
