"""
Retrieval-Augmented Generation (RAG) core.

A small, dependency-light RAG layer over the `knowledge_base/` docs:
  * chunk_text  — split a document into overlapping chunks (pure, testable)
  * cosine_topk — top-K cosine similarity search over a matrix (pure, testable)
  * embed       — Azure OpenAI embeddings for a list of texts
  * build_store — chunk + embed the knowledge base, persist to SQLite
  * load_store / search — read the store and retrieve top-K chunks for a query

The store is a SQLite DB with one row per chunk: (id, source, chunk, embedding).
Embeddings are stored as float32 bytes. The knowledge base is small, so search
loads all vectors and does an exact cosine scan — no ANN index needed.
"""

import os
import sqlite3
from pathlib import Path

import numpy as np

DEFAULT_DB = Path(__file__).resolve().parent.parent / "rag" / "knowledge.db"


# --------------------------------------------------------------------------- #
# Pure helpers (no Azure / no I/O) — unit-tested in CI
# --------------------------------------------------------------------------- #
def chunk_text(text, max_chars=650, overlap=120):
    """Split text into overlapping chunks, preferring paragraph boundaries."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = f"{buf}\n\n{p}" if buf else p
        else:
            if buf:
                chunks.append(buf)
            # If a single paragraph is huge, hard-split it with overlap.
            if len(p) > max_chars:
                for i in range(0, len(p), max_chars - overlap):
                    chunks.append(p[i : i + max_chars])
                buf = ""
            else:
                buf = p
    if buf:
        chunks.append(buf)
    return chunks


def cosine_topk(query_vec, matrix, k=3):
    """Return [(index, score), ...] of the top-k rows by cosine similarity."""
    q = np.asarray(query_vec, dtype=np.float32)
    m = np.asarray(matrix, dtype=np.float32)
    qn = q / (np.linalg.norm(q) + 1e-9)
    mn = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
    sims = mn @ qn
    idx = np.argsort(sims)[::-1][:k]
    return [(int(i), float(sims[i])) for i in idx]


# --------------------------------------------------------------------------- #
# Azure OpenAI embeddings
# --------------------------------------------------------------------------- #
_client = None


def _embed_client():
    global _client
    if _client is None:
        from openai import AzureOpenAI

        _client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )
    return _client


def embed(texts):
    """Embed a list of texts with the configured Azure OpenAI embedding model."""
    deployment = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    resp = _embed_client().embeddings.create(model=deployment, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)


# --------------------------------------------------------------------------- #
# SQLite store
# --------------------------------------------------------------------------- #
def build_store(kb_dir, db_path=DEFAULT_DB):
    """Chunk + embed every markdown file in kb_dir and write the SQLite store."""
    kb_dir, db_path = Path(kb_dir), Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []  # (source, chunk)
    for md in sorted(kb_dir.glob("*.md")):
        for ch in chunk_text(md.read_text()):
            rows.append((md.name, ch))

    vectors = embed([c for _, c in rows])

    con = sqlite3.connect(db_path)
    con.execute("DROP TABLE IF EXISTS chunks")
    con.execute("CREATE TABLE chunks (id INTEGER PRIMARY KEY, source TEXT, chunk TEXT, embedding BLOB)")
    con.executemany(
        "INSERT INTO chunks (source, chunk, embedding) VALUES (?, ?, ?)",
        [(src, ch, vectors[i].tobytes()) for i, (src, ch) in enumerate(rows)],
    )
    con.commit()
    con.close()
    return len(rows)


def load_store(db_path=DEFAULT_DB):
    """Load (sources, chunks, matrix) from the SQLite store."""
    con = sqlite3.connect(str(db_path))
    cur = con.execute("SELECT source, chunk, embedding FROM chunks ORDER BY id")
    sources, chunks, vecs = [], [], []
    for source, chunk, emb in cur.fetchall():
        sources.append(source)
        chunks.append(chunk)
        vecs.append(np.frombuffer(emb, dtype=np.float32))
    con.close()
    return sources, chunks, np.vstack(vecs) if vecs else np.empty((0, 0))


def search(query, db_path=DEFAULT_DB, top_k=3):
    """Retrieve the top-K knowledge chunks for a query (with source + score)."""
    sources, chunks, matrix = load_store(db_path)
    if len(chunks) == 0:
        return []
    qvec = embed([query])[0]
    return [
        {"source": sources[i], "chunk": chunks[i], "score": round(score, 3)}
        for i, score in cosine_topk(qvec, matrix, top_k)
    ]
