from app.agent import Agent
from app.llm import MockLLM
from app.main import load_index
from app.schemas import LLMMessage, ToolCall
from app.tools import ToolRegistry


def make_agent(llm=None, max_steps=4):
    return Agent(llm or MockLLM(), ToolRegistry(load_index()), max_steps=max_steps)


def test_answers_with_grounded_citation():
    res = make_agent().run("How long do customers have to request a full refund?")
    assert "30 days" in res.answer
    assert res.citations == ["refund-policy"]
    assert res.tools_used == ["search_docs"]
    assert res.llm_calls == 2


def test_uses_calculator_for_math():
    res = make_agent().run("What is 600 * 12?")
    assert res.tools_used == ["calculator"]
    assert "7200" in res.answer


def test_unanswerable_question_admits_ignorance():
    res = make_agent().run("Which cloud region hosts the EU data center?")
    assert "don't know" in res.answer
    assert res.citations == []


class Hallucinator:
    """Cites a document that was never retrieved; the agent must drop it."""

    def complete(self, messages, tools):
        if any(m["role"] == "tool" for m in messages):
            return LLMMessage(content="Totally made up. [ghost-doc]")
        return LLMMessage(tool_calls=[ToolCall(id="1", name="search_docs", arguments={"query": "refund"})])


def test_fabricated_citations_are_filtered():
    assert make_agent(Hallucinator()).run("refund question").citations == []


class Looper:
    def complete(self, messages, tools):
        return LLMMessage(tool_calls=[ToolCall(id="1", name="calculator", arguments={"expression": "1+1"})])


def test_step_limit_stops_runaway_loops():
    res = make_agent(Looper(), max_steps=3).run("loop forever please")
    assert "step limit" in res.answer
    assert res.llm_calls == 3


class BadArgs:
    def __init__(self):
        self.turn = 0

    def complete(self, messages, tools):
        self.turn += 1
        if self.turn == 1:
            return LLMMessage(tool_calls=[ToolCall(id="1", name="search_docs", arguments={"q": "x"})])
        return LLMMessage(content="recovered")


def test_tool_errors_are_fed_back_not_raised():
    res = make_agent(BadArgs()).run("bad tool args")
    assert res.answer == "recovered"
    assert res.steps[1].detail["result"]["error"] == "invalid arguments"
