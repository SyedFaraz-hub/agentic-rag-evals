from app.main import load_index
from app.retrieval import HybridIndex, chunk_text, tokenize


def test_tokenize_drops_stopwords():
    assert tokenize("How is the data encrypted?") == ["data", "encrypted"]


def test_chunking_overlaps_long_paragraphs():
    text = " ".join(f"w{i}" for i in range(150))
    chunks = chunk_text("d", text, max_words=60, overlap=10)
    assert len(chunks) == 3
    assert chunks[0].text.split()[-10:] == chunks[1].text.split()[:10]


def test_search_finds_relevant_doc():
    idx = load_index()
    hits = idx.search("what is the free plan rate limit", k=3)
    assert hits and hits[0].chunk.doc_id == "api-limits"


def test_unrelated_query_returns_nothing():
    assert load_index().search("quantum chromodynamics lattice", k=3) == []


def test_empty_index():
    assert HybridIndex().search("anything") == []
