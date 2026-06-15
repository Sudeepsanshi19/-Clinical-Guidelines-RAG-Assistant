from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from clinical_rag.pipeline import answer_query


st.set_page_config(page_title="Clinical Guidelines RAG Assistant", layout="wide")

st.title("Clinical Guidelines RAG Assistant")

with st.sidebar:
    strategy = st.radio("Retrieval strategy", ["hybrid", "dense", "bm25"], index=0)
    top_k = st.slider("Retrieved chunks", min_value=3, max_value=10, value=5)
    use_ollama = st.toggle("Use Ollama if configured", value=True)

question = st.text_area(
    "Clinical guideline question",
    placeholder="Example: What does WHO recommend for advanced HIV disease management?",
    height=100,
)

if st.button("Search guidelines", type="primary") and question.strip():
    try:
        result = answer_query(question.strip(), strategy=strategy, top_k=top_k, prefer_ollama=use_ollama)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Citations")
    for citation in result["citations"]:
        with st.expander(
            f"[{citation['ref']}] {citation['guideline']} - p. {citation['page']}",
            expanded=citation["ref"] == 1,
        ):
            st.write(f"Source: {citation['source']} ({citation['year']})")
            st.write(f"Section: {citation['section']}")
            st.write(f"Score: {citation['score']}")
            st.link_button("Open source", citation["url"])
            st.caption(result["contexts"][citation["ref"] - 1])

