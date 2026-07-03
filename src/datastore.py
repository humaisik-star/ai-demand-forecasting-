"""
Read an analysis table from SQLite when available, otherwise fall back to CSV.

This keeps the CSV flow intact — the SQLite DB (db/analysis.db, built by
build_db.py) is an *extra* layer. If the DB and table exist they are preferred;
if anything is missing the CSV is read directly, so nothing breaks.
"""

import sqlite3
from pathlib import Path

import pandas as pd

DEFAULT_DB = Path(__file__).resolve().parent.parent / "db" / "analysis.db"


def load(table, csv_path, db_path=DEFAULT_DB):
    """Return a DataFrame for `table` from the DB, else read `csv_path`."""
    db_path = Path(db_path)
    if db_path.exists():
        try:
            con = sqlite3.connect(db_path)
            df = pd.read_sql_query(f'SELECT * FROM "{table}"', con)
            con.close()
            return df
        except Exception:
            pass  # table missing / DB unreadable -> fall back to CSV
    return pd.read_csv(csv_path)
