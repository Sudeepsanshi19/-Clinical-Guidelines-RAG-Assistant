from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
EVAL_DIR = DATA_DIR / "eval"
STORAGE_DIR = ROOT_DIR / "storage"
MANIFEST_PATH = DATA_DIR / "guidelines_manifest.csv"
CHUNKS_PATH = STORAGE_DIR / "chunks.jsonl"
INDEX_PATH = STORAGE_DIR / "rag_index.pkl"
EVAL_RESULTS_PATH = STORAGE_DIR / "evaluation_results.csv"

