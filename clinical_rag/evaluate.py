from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from clinical_rag.config import EVAL_DIR, EVAL_RESULTS_PATH
from clinical_rag.pipeline import answer_query
from clinical_rag.text import tokenize


def load_scenarios(path: Path = EVAL_DIR / "clinical_scenarios.jsonl") -> list[dict]:
    scenarios = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                scenarios.append(json.loads(line))
    return scenarios


def local_faithfulness(answer: str, contexts: list[str]) -> float:
    answer_terms = set(tokenize(answer))
    if not answer_terms:
        return 0.0
    context_terms = set(tokenize(" ".join(contexts)))
    return round(len(answer_terms & context_terms) / len(answer_terms), 4)


def local_answer_relevance(question: str, answer: str) -> float:
    question_terms = set(tokenize(question))
    answer_terms = set(tokenize(answer))
    if not question_terms:
        return 0.0
    return round(len(question_terms & answer_terms) / len(question_terms), 4)


def run_evaluation(strategies: list[str] | None = None, top_k: int = 5) -> pd.DataFrame:
    strategies = strategies or ["dense", "hybrid"]
    rows = []
    for scenario in load_scenarios():
        for strategy in strategies:
            result = answer_query(scenario["question"], strategy=strategy, top_k=top_k, prefer_ollama=False)
            rows.append(
                {
                    "id": scenario["id"],
                    "question": scenario["question"],
                    "expected_topic": scenario.get("expected_topic", ""),
                    "strategy": strategy,
                    "answer": result["answer"],
                    "citations": json.dumps(result["citations"], ensure_ascii=False),
                    "contexts": json.dumps(result["contexts"], ensure_ascii=False),
                    "faithfulness": local_faithfulness(result["answer"], result["contexts"]),
                    "answer_relevance": local_answer_relevance(scenario["question"], result["answer"]),
                }
            )
    frame = pd.DataFrame(rows)
    EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(EVAL_RESULTS_PATH, index=False)
    return frame

