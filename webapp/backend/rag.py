"""RAG retrieval for the web backend (search-time only).

Reads the bundled SQLite vector store and does an exact cosine top-K scan.
Query embedding uses Azure OpenAI (same resource as chat).
"""

import os
import sqlite3
from pathlib import Path

import numpy as np

DB_PATH = Path(__file__).parent / "knowledge.db"

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
    deployment = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    resp = _embed_client().embeddings.create(model=deployment, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)


def cosine_topk(query_vec, matrix, k=3):
    q = np.asarray(query_vec, dtype=np.float32)
    m = np.asarray(matrix, dtype=np.float32)
    qn = q / (np.linalg.norm(q) + 1e-9)
    mn = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
    sims = mn @ qn
    idx = np.argsort(sims)[::-1][:k]
    return [(int(i), float(sims[i])) for i in idx]


def load_store(db_path=DB_PATH):
    con = sqlite3.connect(str(db_path))
    cur = con.execute("SELECT source, chunk, embedding FROM chunks ORDER BY id")
    sources, chunks, vecs = [], [], []
    for source, chunk, emb in cur.fetchall():
        sources.append(source)
        chunks.append(chunk)
        vecs.append(np.frombuffer(emb, dtype=np.float32))
    con.close()
    return sources, chunks, np.vstack(vecs) if vecs else np.empty((0, 0))


def search(query, top_k=3):
    sources, chunks, matrix = load_store()
    if len(chunks) == 0:
        return []
    qvec = embed([query])[0]
    return [
        {"source": sources[i], "chunk": chunks[i], "score": round(score, 3)}
        for i, score in cosine_topk(qvec, matrix, top_k)
    ]
