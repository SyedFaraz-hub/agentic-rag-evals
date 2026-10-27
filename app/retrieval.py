"""Hybrid retrieval: BM25 keyword ranking + hashed-embedding cosine similarity,
merged with Reciprocal Rank Fusion (RRF)."""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import numpy as np

from .schemas import Chunk, Hit

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "is", "are", "of", "to", "in", "and", "or", "for", "on",
    "with", "how", "what", "do", "does", "can", "i", "my", "we", "our", "it",
    "be", "by", "at", "as", "this", "that", "from", "your",
}


def _stem(tok: str) -> str:
    """Tiny plural stemmer: enough for refunds/refund, requests/request."""
    return tok[:-1] if len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss") else tok


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def chunk_text(doc_id: str, text: str, max_words: int = 60, overlap: int = 10) -> list[Chunk]:
    """Split on paragraphs, then window long paragraphs with overlap."""
    chunks: list[Chunk] = []
    n = 0
    prefix = ""
    for para in [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]:
        if para.startswith("#"):  # keep the heading as context for the next paragraph
            prefix = para.lstrip("# ").strip() + ". "
            continue
        words = (prefix + para).split()
        prefix = ""
        start = 0
        while start < len(words):
            piece = words[start : start + max_words]
            chunks.append(Chunk(chunk_id=f"{doc_id}#{n}", doc_id=doc_id, text=" ".join(piece)))
            n += 1
            if start + max_words >= len(words):
                break
            start += max_words - overlap
    return chunks


def hash_embed(text: str, dim: int = 256) -> np.ndarray:
    """Deterministic feature-hashing embedding (no model download, offline-safe).
    Swap for a real embedding model by replacing this function."""
    vec = np.zeros(dim, dtype=np.float32)
    toks = tokenize(text)
    for tok in toks + [a + "_" + b for a, b in zip(toks, toks[1:], strict=False)]:
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0 if (h >> 8) & 1 else -1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


class HybridIndex:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.chunks: list[Chunk] = []
        self._tf: list[Counter[str]] = []
        self._df: Counter[str] = Counter()
        self._len: list[int] = []
        self._emb: list[np.ndarray] = []

    def add_document(self, doc_id: str, text: str) -> int:
        new = chunk_text(doc_id, text)
        for c in new:
            toks = tokenize(c.text)
            tf = Counter(toks)
            self.chunks.append(c)
            self._tf.append(tf)
            self._len.append(len(toks))
            self._df.update(tf.keys())
            self._emb.append(hash_embed(c.text))
        return len(new)

    def _bm25(self, query: str) -> np.ndarray:
        n = len(self.chunks)
        scores = np.zeros(n)
        if not n:
            return scores
        avg = (sum(self._len) / n) or 1.0
        for term in set(tokenize(query)):
            df = self._df.get(term, 0)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, tf in enumerate(self._tf):
                f = tf.get(term, 0)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * self._len[i] / avg)
                    scores[i] += idf * f * (self.k1 + 1) / denom
        return scores

    def search(self, query: str, k: int = 3) -> list[Hit]:
        if not self.chunks:
            return []
        bm25 = self._bm25(query)
        q = hash_embed(query)
        dense = np.array([float(q @ e) for e in self._emb])
        fused: dict[int, float] = {}
        for ranking in (np.argsort(-bm25), np.argsort(-dense)):
            for rank, idx in enumerate(ranking[: max(k * 3, 10)]):
                if bm25[idx] > 0 or dense[idx] > 0.15:
                    fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (60 + rank + 1)
        order = sorted(fused, key=fused.get, reverse=True)[:k]
        # Require at least one lexical match so unrelated queries return nothing.
        return [Hit(chunk=self.chunks[i], score=round(fused[i], 5)) for i in order if bm25[i] > 0]
