"""CI-safe tests for the RAG core (pure functions — no Azure calls)."""

import numpy as np

from src.rag import chunk_text, cosine_topk


def test_chunk_text_nonempty_and_bounded():
    text = "\n\n".join(f"Paragraf {i}. " + "kelime " * 40 for i in range(6))
    chunks = chunk_text(text, max_chars=300, overlap=50)
    assert len(chunks) >= 2
    assert all(c.strip() for c in chunks)
    # No chunk vastly exceeds the cap (paragraphs pack up to max_chars).
    assert max(len(c) for c in chunks) <= 300 * 1.5


def test_chunk_text_hard_splits_huge_paragraph():
    huge = "x" * 2000
    chunks = chunk_text(huge, max_chars=500, overlap=100)
    assert len(chunks) >= 4                      # 2000 chars can't fit in one 500 cap
    assert all(len(c) <= 500 for c in chunks)


def test_cosine_topk_orders_by_similarity():
    matrix = np.array([[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]], dtype=float)
    q = np.array([1, 0, 0], dtype=float)
    top = cosine_topk(q, matrix, k=3)
    idx = [i for i, _ in top]
    assert idx[0] == 0                            # identical vector first
    assert idx[1] == 2                            # near-parallel next
    assert np.isclose(top[0][1], 1.0, atol=1e-4)  # perfect match ~1.0


def test_cosine_topk_respects_k():
    matrix = np.random.RandomState(0).rand(10, 5)
    q = np.random.RandomState(1).rand(5)
    assert len(cosine_topk(q, matrix, k=3)) == 3
    # Scores are sorted descending.
    scores = [s for _, s in cosine_topk(q, matrix, k=10)]
    assert scores == sorted(scores, reverse=True)
