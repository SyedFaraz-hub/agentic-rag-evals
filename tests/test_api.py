from fastapi.testclient import TestClient

from app.llm import MockLLM
from app.main import create_app

client = TestClient(create_app(llm=MockLLM()))


def test_health():
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["chunks"] >= 5


def test_ask_returns_structured_answer():
    r = client.post("/ask", json={"question": "What hours is the support team available?"})
    assert r.status_code == 200
    body = r.json()
    assert body["citations"] == ["support-hours"]
    assert body["latency_ms"] >= 0


def test_ask_validates_input():
    assert client.post("/ask", json={"question": "hi"}).status_code == 422


def test_ingest_then_ask_new_document():
    r = client.post("/ingest", json={"doc_id": "pets", "text": "Our office allows dogs on Fridays."})
    assert r.json()["chunks_added"] == 1
    body = client.post("/ask", json={"question": "Are dogs allowed in the office?"}).json()
    assert body["citations"] == ["pets"]


def test_stream_emits_sse_events_in_order():
    with client.stream("POST", "/ask/stream", json={"question": "How is data encrypted at rest?"}) as r:
        text = "".join(r.iter_text())
    events = [line.split(": ")[1] for line in text.splitlines() if line.startswith("event:")]
    assert events == ["tool_call", "tool_result", "final"]
