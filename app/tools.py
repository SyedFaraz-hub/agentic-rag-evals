"""Tool registry. Each tool declares a JSON schema (sent to the model) and a
Pydantic model that validates the model-produced arguments before execution."""
from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .retrieval import HybridIndex


class SearchArgs(BaseModel):
    query: str = Field(min_length=2)
    k: int = Field(default=3, ge=1, le=8)


class CalcArgs(BaseModel):
    expression: str = Field(min_length=1, max_length=200)


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def safe_eval(expr: str) -> float:
    """Evaluate arithmetic only (no names, calls, or attribute access)."""

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.operand))
        raise ValueError("unsupported expression")

    return walk(ast.parse(expr, mode="eval"))


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    run: Callable[[BaseModel], dict[str, Any]]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }


class ToolRegistry:
    def __init__(self, index: HybridIndex) -> None:
        self.index = index
        self.tools: dict[str, Tool] = {
            "search_docs": Tool(
                "search_docs",
                "Search the knowledge base. Returns the most relevant passages with ids.",
                SearchArgs,
                self._search,
            ),
            "calculator": Tool(
                "calculator", "Evaluate an arithmetic expression.", CalcArgs, self._calc
            ),
        }

    def _search(self, a: SearchArgs) -> dict[str, Any]:  # type: ignore[override]
        hits = self.index.search(a.query, a.k)
        return {
            "results": [
                {"doc_id": h.chunk.doc_id, "chunk_id": h.chunk.chunk_id,
                 "text": h.chunk.text, "score": h.score}
                for h in hits
            ]
        }

    def _calc(self, a: CalcArgs) -> dict[str, Any]:  # type: ignore[override]
        return {"value": safe_eval(a.expression)}

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self.tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Never raises: errors are returned to the model so it can recover."""
        tool = self.tools.get(name)
        if tool is None:
            return {"error": f"unknown tool '{name}'"}
        try:
            return tool.run(tool.args_model.model_validate(arguments))
        except ValidationError as e:
            return {"error": "invalid arguments", "details": e.errors(include_url=False)}
        except (ValueError, ZeroDivisionError, SyntaxError) as e:
            return {"error": str(e)}
