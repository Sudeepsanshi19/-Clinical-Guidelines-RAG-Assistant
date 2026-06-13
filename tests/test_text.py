from clinical_rag.text import chunk_words, tokenize


def test_tokenize_removes_common_question_words():
    assert "recommended" not in tokenize("What is recommended for advanced HIV disease?")
    assert "advanced" in tokenize("What is recommended for advanced HIV disease?")


def test_chunk_words_uses_overlap():
    chunks = list(chunk_words(["a", "b", "c", "d", "e"], size=3, overlap=1))
    assert chunks == [["a", "b", "c"], ["c", "d", "e"]]

