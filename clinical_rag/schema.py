from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class GuidelineDoc:
    doc_id: str
    title: str
    source: str
    year: str
    topic: str
    publication_url: str
    pdf_url: str
    local_path: str


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    source: str
    year: str
    topic: str
    section: str
    page: int
    text: str
    source_url: str
    extra: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class SearchHit:
    chunk: Chunk
    score: float
    strategy: str

