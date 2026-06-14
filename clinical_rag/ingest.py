from __future__ import annotations

import csv
import json
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from clinical_rag.config import CHUNKS_PATH, MANIFEST_PATH, PDF_DIR
from clinical_rag.schema import Chunk, GuidelineDoc
from clinical_rag.text import chunk_words, likely_heading, normalize_whitespace


def load_manifest(path: Path = MANIFEST_PATH) -> list[GuidelineDoc]:
    docs: list[GuidelineDoc] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            doc_id = row["doc_id"]
            docs.append(
                GuidelineDoc(
                    doc_id=doc_id,
                    title=row["title"],
                    source=row["source"],
                    year=row["year"],
                    topic=row["topic"],
                    publication_url=row["publication_url"],
                    pdf_url=row["pdf_url"],
                    local_path=str(PDF_DIR / f"{doc_id}.pdf"),
                )
            )
    return docs


def page_section(text: str, previous: str) -> str:
    for raw_line in text.splitlines():
        line = normalize_whitespace(raw_line)
        if likely_heading(line):
            return line[:120]
    return previous or "Unknown section"


def extract_chunks(
    doc: GuidelineDoc,
    chunk_size: int = 260,
    overlap: int = 45,
) -> list[Chunk]:
    pdf_path = Path(doc.local_path)
    if not pdf_path.exists():
        return []

    try:
        reader = PdfReader(str(pdf_path))
    except (PdfReadError, OSError) as exc:
        print(f"Skipping unreadable PDF {pdf_path.name}: {exc}")
        return []
    chunks: list[Chunk] = []
    section = "Unknown section"

    for page_index, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        if not raw_text.strip():
            continue
        section = page_section(raw_text, section)
        words = normalize_whitespace(raw_text).split()
        for chunk_index, piece in enumerate(chunk_words(words, chunk_size, overlap), start=1):
            text = " ".join(piece)
            chunk_id = f"{doc.doc_id}:p{page_index}:c{chunk_index}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    title=doc.title,
                    source=doc.source,
                    year=doc.year,
                    topic=doc.topic,
                    section=section,
                    page=page_index,
                    text=text,
                    source_url=doc.pdf_url or doc.publication_url,
                )
            )
    return chunks


def chunk_to_dict(chunk: Chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "title": chunk.title,
        "source": chunk.source,
        "year": chunk.year,
        "topic": chunk.topic,
        "section": chunk.section,
        "page": chunk.page,
        "text": chunk.text,
        "source_url": chunk.source_url,
        "extra": chunk.extra,
    }


def dict_to_chunk(data: dict) -> Chunk:
    return Chunk(
        chunk_id=data["chunk_id"],
        doc_id=data["doc_id"],
        title=data["title"],
        source=data["source"],
        year=str(data["year"]),
        topic=data["topic"],
        section=data["section"],
        page=int(data["page"]),
        text=data["text"],
        source_url=data["source_url"],
        extra=data.get("extra", {}),
    )


def write_chunks(chunks: list[Chunk], path: Path = CHUNKS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk_to_dict(chunk), ensure_ascii=False) + "\n")


def read_chunks(path: Path = CHUNKS_PATH) -> list[Chunk]:
    if not path.exists():
        return []
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunks.append(dict_to_chunk(json.loads(line)))
    return chunks


def ingest_all(manifest_path: Path = MANIFEST_PATH) -> list[Chunk]:
    docs = load_manifest(manifest_path)
    all_chunks: list[Chunk] = []
    missing: list[str] = []
    for doc in docs:
        chunks = extract_chunks(doc)
        if chunks:
            all_chunks.extend(chunks)
        else:
            missing.append(doc.doc_id)
    write_chunks(all_chunks)
    if missing:
        print("Missing or unreadable PDFs:", ", ".join(missing))
    print(f"Wrote {len(all_chunks)} chunks to {CHUNKS_PATH}")
    return all_chunks
