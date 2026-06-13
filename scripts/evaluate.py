from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinical_rag.config import EVAL_RESULTS_PATH
from clinical_rag.evaluate import run_evaluation


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    frame = run_evaluation(strategies=["dense", "hybrid"])
    summary = frame.groupby("strategy")[["faithfulness", "answer_relevance"]].mean().round(4)
    print(summary)
    print(f"\nSaved detailed results to {EVAL_RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
