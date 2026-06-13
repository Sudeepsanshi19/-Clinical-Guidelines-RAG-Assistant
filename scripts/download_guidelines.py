from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinical_rag.config import PDF_DIR
from clinical_rag.ingest import load_manifest


def is_valid_pdf(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 4 and path.read_bytes()[:4] == b"%PDF"


def download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "clinical-guidelines-rag/0.1"})
    with urllib.request.urlopen(request, timeout=90) as response:
        target.write_bytes(response.read())
    if not is_valid_pdf(target):
        preview = target.read_bytes()[:80].decode("latin1", errors="replace")
        target.unlink(missing_ok=True)
        raise ValueError(f"download did not return a PDF; first bytes: {preview!r}")


def main() -> int:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    docs = load_manifest()
    downloaded = 0
    skipped = 0
    failed: list[str] = []

    for doc in docs:
        target = PDF_DIR / f"{doc.doc_id}.pdf"
        if is_valid_pdf(target):
            skipped += 1
            continue
        try:
            print(f"Downloading {doc.doc_id}...")
            download(doc.pdf_url, target)
            downloaded += 1
        except Exception as exc:
            failed.append(f"{doc.doc_id}: {exc}")

    print(f"Downloaded: {downloaded}; skipped: {skipped}; failed: {len(failed)}")
    if failed:
        print("Failures:")
        for item in failed:
            print(f"  - {item}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
