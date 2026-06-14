from __future__ import annotations

import hashlib
import math
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from clinical_rag.config import INDEX_PATH
from clinical_rag.ingest import read_chunks
from clinical_rag.schema import Chunk, SearchHit
from clinical_rag.text import tokenize


def _stable_hash(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


class HashingEmbedder:
    def __init__(self, dims: int = 384) -> None:
        self.dims = dims

    def encode_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dims, dtype=np.float32)
        tokens = tokenize(text)
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        for feature in features:
            hashed = _stable_hash(feature)
            index = hashed % self.dims
            sign = 1.0 if hashed & 1 else -1.0
            vector[index] += sign
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dims), dtype=np.float32)
        return np.vstack([self.encode_one(text) for text in texts])


@dataclass
class BM25Index:
    tokenized_docs: list[list[str]]
    idf: dict[str, float]
    avgdl: float
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(cls, texts: list[str]) -> "BM25Index":
        tokenized = [tokenize(text) for text in texts]
        doc_count = len(tokenized)
        avgdl = sum(len(doc) for doc in tokenized) / max(doc_count, 1)
        document_frequency: Counter[str] = Counter()
        for doc in tokenized:
            document_frequency.update(set(doc))
        idf = {
            term: math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }
        return cls(tokenized_docs=tokenized, idf=idf, avgdl=avgdl)

    def scores(self, query: str) -> np.ndarray:
        query_terms = tokenize(query)
        scores = np.zeros(len(self.tokenized_docs), dtype=np.float32)
        if not query_terms:
            return scores
        for index, doc in enumerate(self.tokenized_docs):
            freqs = Counter(doc)
            doc_len = len(doc)
            score = 0.0
            for term in query_terms:
                if term not in freqs:
                    continue
                idf = self.idf.get(term, 0.0)
                numerator = freqs[term] * (self.k1 + 1)
                denominator = freqs[term] + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1))
                score += idf * numerator / denominator
            scores[index] = score
        return scores


class RAGIndex:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, bm25: BM25Index) -> None:
        self.chunks = chunks
        self.embedder = HashingEmbedder(dims=embeddings.shape[1] if embeddings.size else 384)
        self.embeddings = embeddings
        self.bm25 = bm25

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "RAGIndex":
        embedder = HashingEmbedder()
        texts = [chunk.text for chunk in chunks]
        embeddings = embedder.encode(texts)
        bm25 = BM25Index.build(texts)
        return cls(chunks=chunks, embeddings=embeddings, bm25=bm25)

    def save(self, path: Path = INDEX_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: Path = INDEX_PATH) -> "RAGIndex":
        with path.open("rb") as handle:
            return pickle.load(handle)

    def dense_scores(self, query: str) -> np.ndarray:
        if len(self.chunks) == 0:
            return np.zeros(0, dtype=np.float32)
        query_embedding = self.embedder.encode_one(query)
        return self.embeddings @ query_embedding

    def search(self, query: str, strategy: str = "hybrid", top_k: int = 5) -> list[SearchHit]:
        strategy = strategy.lower()
        dense = self.dense_scores(query)
        bm25 = self.bm25.scores(query)

        if strategy == "dense":
            scores = dense
        elif strategy == "bm25":
            scores = bm25
        elif strategy == "hybrid":
            scores = 0.55 * _normalize(dense) + 0.45 * _normalize(bm25)
        else:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")

        if len(scores) == 0:
            return []
        scores = scores * self._quality_weights()
        order = np.argsort(scores)[::-1][:top_k]
        return [
            SearchHit(chunk=self.chunks[int(index)], score=float(scores[int(index)]), strategy=strategy)
            for index in order
            if float(scores[int(index)]) > 0
        ]

    def _quality_weights(self) -> np.ndarray:
        weights = np.ones(len(self.chunks), dtype=np.float32)
        for index, chunk in enumerate(self.chunks):
            section = chunk.section.lower()
            text = chunk.text.lower()
            if chunk.page <= 2:
                weights[index] *= 0.35
            if section in {"unknown section", "contents", "abbreviations"}:
                weights[index] *= 0.45
            if "....." in text or text.count(".") > 35:
                weights[index] *= 0.35
            if len(tokenize(chunk.text)) < 30:
                weights[index] *= 0.5
        return weights


def _normalize(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    minimum = float(scores.min())
    maximum = float(scores.max())
    if math.isclose(minimum, maximum):
        return np.zeros_like(scores)
    return (scores - minimum) / (maximum - minimum)


def build_and_save_index(path: Path = INDEX_PATH) -> RAGIndex:
    chunks = read_chunks()
    index = RAGIndex.build(chunks)
    index.save(path)
    print(f"Wrote index for {len(chunks)} chunks to {path}")
    return index
