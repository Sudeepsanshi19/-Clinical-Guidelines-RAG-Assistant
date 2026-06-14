from __future__ import annotations

import re
from collections.abc import Iterable


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+-]*")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

STOPWORDS = {
    "a",
    "about",
    "above",
    "according",
    "after",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "care",
    "clinical",
    "does",
    "for",
    "from",
    "guidance",
    "guideline",
    "guidelines",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "patient",
    "patients",
    "provide",
    "recommended",
    "recommendations",
    "should",
    "that",
    "the",
    "to",
    "use",
    "what",
    "when",
    "which",
    "with",
}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    tokens = [match.group(0).lower() for match in TOKEN_RE.finditer(text)]
    if keep_stopwords:
        return tokens
    return [token for token in tokens if token not in STOPWORDS and len(token) > 2]


def split_sentences(text: str) -> list[str]:
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return []
    return [part.strip() for part in SENTENCE_RE.split(cleaned) if part.strip()]


def chunk_words(words: list[str], size: int, overlap: int) -> Iterable[list[str]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if overlap >= size:
        raise ValueError("chunk overlap must be smaller than chunk size")
    step = size - overlap
    for start in range(0, len(words), step):
        piece = words[start : start + size]
        if piece:
            yield piece
        if start + size >= len(words):
            break


def likely_heading(line: str) -> bool:
    stripped = line.strip()
    if not 4 <= len(stripped) <= 120:
        return False
    if stripped.endswith("."):
        return False
    numbered = re.match(r"^(\d+(\.\d+)*|[A-Z])\s+[\w,;:() -]+$", stripped)
    mostly_title = stripped.istitle() and len(stripped.split()) <= 12
    mostly_upper = stripped.upper() == stripped and any(ch.isalpha() for ch in stripped)
    return bool(numbered or mostly_title or mostly_upper)

