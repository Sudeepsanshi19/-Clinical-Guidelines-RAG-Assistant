from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinical_rag.pipeline import answer_query


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Ask the Clinical Guidelines RAG Assistant a question.")
    parser.add_argument("question", help="Clinical guideline question")
    parser.add_argument("--strategy", choices=["dense", "bm25", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-ollama", action="store_true", help="Use local extractive answer generation only")
    parser.add_argument("--json", action="store_true", help="Print full JSON response")
    args = parser.parse_args()

    result = answer_query(
        args.question,
        strategy=args.strategy,
        top_k=args.top_k,
        prefer_ollama=not args.no_ollama,
    )
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    print(result["answer"])
    print("\nCitations:")
    for citation in result["citations"]:
        print(
            f"[{citation['ref']}] {citation['guideline']} | {citation['section']} | "
            f"p. {citation['page']} | {citation['url']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
