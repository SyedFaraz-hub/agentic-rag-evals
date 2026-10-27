import pytest

from app.main import load_index
from app.tools import ToolRegistry, safe_eval


@pytest.mark.parametrize("expr,expected", [("2+3*4", 14), ("(1+2)**3", 27), ("-5+2", -3), ("10/4", 2.5)])
def test_safe_eval(expr, expected):
    assert safe_eval(expr) == expected


@pytest.mark.parametrize("expr", ["__import__('os').system('ls')", "open('x')", "a+1", "1+"])
def test_safe_eval_rejects_unsafe(expr):
    with pytest.raises((ValueError, SyntaxError)):
        safe_eval(expr)


def test_execute_validates_arguments():
    reg = ToolRegistry(load_index())
    out = reg.execute("search_docs", {"query": "x", "k": 99})
    assert out["error"] == "invalid arguments"


def test_execute_unknown_tool_and_math_errors_are_returned_not_raised():
    reg = ToolRegistry(load_index())
    assert "unknown tool" in reg.execute("nope", {})["error"]
    assert "error" in reg.execute("calculator", {"expression": "1/0"})


def test_schemas_are_openai_function_format():
    schemas = ToolRegistry(load_index()).schemas()
    assert {s["function"]["name"] for s in schemas} == {"search_docs", "calculator"}
    assert all(s["type"] == "function" for s in schemas)
