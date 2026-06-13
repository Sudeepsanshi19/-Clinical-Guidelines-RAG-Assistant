from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinical_rag.ingest import ingest_all
from clinical_rag.retrievers import build_and_save_index


def main() -> int:
    chunks = ingest_all()
    if not chunks:
        print("No chunks were created. Download PDFs first or place CPG PDFs in data/pdfs.")
        return 1
    build_and_save_index()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

