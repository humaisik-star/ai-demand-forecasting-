"""
Build the RAG knowledge base: chunk the knowledge_base/ docs, embed them with
Azure OpenAI, and persist the vectors to rag/knowledge.db (SQLite).

Run:
    .venv/bin/python build_kb.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.rag import DEFAULT_DB, build_store

KB_DIR = Path("knowledge_base")


def main():
    load_dotenv()
    for v in ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"]:
        if not os.environ.get(v):
            print(f"Missing {v}. Copy .env.example to .env and fill it in.")
            sys.exit(1)

    print(f"Building knowledge base from {KB_DIR}/ ...")
    n = build_store(KB_DIR, DEFAULT_DB)
    print(f"Embedded and stored {n} chunks -> {DEFAULT_DB}")


if __name__ == "__main__":
    main()
