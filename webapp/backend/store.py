"""Read an analysis table from the bundled SQLite DB, else fall back to CSV.

The DB (analysis.db, mirror of the result CSVs) is an extra query layer; if it
is missing or a table is absent, the CSV in data/ is read instead.
"""

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent / "analysis.db"


def load(table, csv_path):
    if DB_PATH.exists():
        try:
            con = sqlite3.connect(DB_PATH)
            df = pd.read_sql_query(f'SELECT * FROM "{table}"', con)
            con.close()
            return df
        except Exception:
            pass
    return pd.read_csv(csv_path)
