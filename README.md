# Clinical Guidelines RAG Assistant

Evidence-focused retrieval augmented generation for clinical practice guidelines (CPGs) from WHO, ICMR, and journal-style sources.

This project is designed as a clinical guideline navigation assistant, not a diagnosis or treatment replacement. Answers are generated only from retrieved guideline chunks and include guideline, section, page, and source citations.

## What Is Included

- PDF downloader driven by a 10-document guideline manifest.
- PDF ingestion with page-aware chunking and metadata tagging.
- Dense retrieval using deterministic local hashing embeddings.
- BM25 retrieval and hybrid BM25+dense ranking.
- CLI query interface with cited answers.
- Streamlit UI scaffold.
- 15 clinical scenario evaluation set.
- Evaluation harness with RAGAS-ready fields plus local fallback metrics.

## Project Layout

```text
app/streamlit_app.py                 Streamlit query UI
data/guidelines_manifest.csv         10 guideline source manifest
data/eval/clinical_scenarios.jsonl   15 evaluation queries
data/pdfs/                           downloaded PDFs
scripts/download_guidelines.py       downloads PDFs from manifest
scripts/ingest_guidelines.py         builds the searchable index
scripts/query.py                     command-line query entrypoint
scripts/evaluate.py                  retrieval/answer evaluation
src/clinical_rag/                    reusable RAG package
storage/                             generated chunks and indexes
```

## Quick Start

Use a virtual environment if possible, then install dependencies:

```powershell
pip install -r requirements.txt
```

Download the starter WHO guideline PDFs:

```powershell
python scripts/download_guidelines.py
```

Build the index:

```powershell
python scripts/ingest_guidelines.py
```

Ask a question:

```powershell
python scripts/query.py "What does the guideline say about advanced HIV disease management?"
```

Compare dense and hybrid retrieval on the 15 scenarios:

```powershell
python scripts/evaluate.py
```

Run the UI:

```powershell
streamlit run app/streamlit_app.py
```

## Retrieval Strategies

The project compares:

- `dense`: cosine search over local dense hashing embeddings.
- `bm25`: lexical BM25.
- `hybrid`: normalized dense + BM25 scores.

The hashing embedder keeps the project runnable without downloading embedding models. For a stronger final submission, replace it with `sentence-transformers` or a LangChain embedding provider.

## Evaluation

The evaluation harness emits per-query rows with:

- question
- strategy
- answer
- retrieved contexts
- citations
- faithfulness score
- answer relevance score

If `ragas` is installed and configured with an LLM/embedding backend, the exported fields can be used directly for RAGAS evaluation. The default local metrics are lightweight heuristics so the pipeline remains testable offline.

## Clinical Safety Guardrails

- The assistant should not answer beyond retrieved guideline evidence.
- Every recommendation should include source citations.
- If retrieved context is weak, the assistant should say so.
- Conflicting sources should be surfaced rather than collapsed into a single unsupported answer.
- Current guideline versions must be verified before clinical use.

