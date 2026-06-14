from __future__ import annotations

from pathlib import Path

from clinical_rag.config import INDEX_PATH
from clinical_rag.generator import answer
from clinical_rag.retrievers import RAGIndex


def load_index(index_path: Path = INDEX_PATH) -> RAGIndex:
    if not index_path.exists():
        raise FileNotFoundError(
            f"No index found at {index_path}. Run scripts/download_guidelines.py and scripts/ingest_guidelines.py first."
        )
    return RAGIndex.load(index_path)


def answer_query(question: str, strategy: str = "hybrid", top_k: int = 5, prefer_ollama: bool = True) -> dict:
    index = load_index()
    hits = index.search(question, strategy=strategy, top_k=top_k)
    payload = answer(question, hits, prefer_ollama=prefer_ollama)
    payload["question"] = question
    payload["strategy"] = strategy
    return payload

