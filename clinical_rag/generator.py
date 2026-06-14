from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from clinical_rag.schema import SearchHit
from clinical_rag.text import split_sentences, tokenize


SYSTEM_INSTRUCTION = """You are a clinical guideline retrieval assistant.
Answer only from the supplied guideline context.
Do not invent doses, diagnoses, or treatment recommendations.
If the context is insufficient, say that the retrieved guideline evidence is insufficient.
Include bracket citations like [1] that refer to the provided sources."""


def build_context(hits: list[SearchHit], max_chars: int = 7000) -> str:
    blocks: list[str] = []
    used = 0
    for i, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        citation = (
            f"[{i}] {chunk.title} | {chunk.source} | {chunk.year} | "
            f"{chunk.section} | p. {chunk.page} | {chunk.source_url}"
        )
        block = f"{citation}\n{chunk.text}"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def cite_hits(hits: list[SearchHit]) -> list[dict[str, str | int | float]]:
    citations = []
    for i, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        citations.append(
            {
                "ref": i,
                "guideline": chunk.title,
                "source": chunk.source,
                "year": chunk.year,
                "section": chunk.section,
                "page": chunk.page,
                "url": chunk.source_url,
                "score": round(hit.score, 4),
            }
        )
    return citations


def generate_with_ollama(question: str, hits: list[SearchHit]) -> str | None:
    model = os.getenv("OLLAMA_MODEL")
    if not model:
        return None
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    prompt = (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Question: {question}\n\n"
        f"Guideline context:\n{build_context(hits)}\n\n"
        "Answer:"
    )
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))
            return (data.get("response") or "").strip() or None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def generate_extractive_answer(question: str, hits: list[SearchHit], max_sentences: int = 4) -> str:
    if not hits:
        return (
            "I could not find sufficiently relevant guideline evidence in the indexed corpus. "
            "Please add or re-index the relevant CPG documents before using this for clinical decision support."
        )

    query_terms = set(tokenize(question))
    candidates: list[tuple[float, int, str]] = []
    for ref, hit in enumerate(hits, start=1):
        for sentence in split_sentences(hit.chunk.text):
            if _low_value_sentence(sentence):
                continue
            sentence_terms = set(tokenize(sentence))
            if not sentence_terms:
                continue
            overlap = len(query_terms & sentence_terms)
            score = overlap + (hit.score * 0.25) - (ref * 0.02)
            candidates.append((score, ref, sentence))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[str] = []
    seen: set[str] = set()
    for score, ref, sentence in candidates:
        normalized = sentence.lower()
        if score <= 0 or normalized in seen:
            continue
        selected.append(f"{sentence} [{ref}]")
        seen.add(normalized)
        if len(selected) >= max_sentences:
            break

    if not selected:
        return (
            "The retrieved guideline sections may be relevant, but they do not contain enough directly matching "
            "evidence to answer the question confidently. Review the cited sources before applying any recommendation."
        )

    prefix = "Based on the retrieved guideline sections: "
    return prefix + " ".join(selected)


def _low_value_sentence(sentence: str) -> bool:
    stripped = sentence.strip()
    if len(stripped) < 45:
        return True
    if "....." in stripped:
        return True
    if stripped.count(".") > 10:
        return True
    if sum(char.isdigit() for char in stripped) > len(stripped) * 0.25:
        return True
    return False


def answer(question: str, hits: list[SearchHit], prefer_ollama: bool = True) -> dict:
    generated = generate_with_ollama(question, hits) if prefer_ollama else None
    if not generated:
        generated = generate_extractive_answer(question, hits)
    return {"answer": generated, "citations": cite_hits(hits), "contexts": [hit.chunk.text for hit in hits]}
