"""
Load the analysis result CSVs into a single SQLite database — an extra query
layer on top of the existing CSV flow. The CSVs remain the source of truth
(the scripts still write them); this just mirrors them into db/analysis.db so
the dashboard, chatbot tools, and API can query them with SQL.

Run (after predict.py / stock.py / inventory_analytics.py / advanced_analytics.py):
    .venv/bin/python build_db.py
"""

import sqlite3
from pathlib import Path

import pandas as pd

RESULTS = Path("results")
DB_PATH = Path("db/analysis.db")

# table name -> source CSV
TABLES = {
    "predictions": "predictions.csv",
    "stock_recommendations": "stock_recommendations.csv",
    "inventory_analytics": "inventory_analytics.csv",
    "advanced_analytics": "advanced_analytics.csv",
    "anomalies": "anomalies.csv",
    "model_metrics": "model_metrics.csv",
    "backtest_metrics": "backtest_metrics.csv",
    "quantile_metrics": "quantile_metrics.csv",
    "abc_xyz_matrix": "abc_xyz_matrix.csv",
}


def build(results=RESULTS, db_path=DB_PATH):
    """Write every available result CSV into db_path as its own table."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    written = {}
    for table, fname in TABLES.items():
        p = Path(results) / fname
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df.to_sql(table, con, if_exists="replace", index=False)
        written[table] = len(df)
    con.commit()
    con.close()
    return written


def main():
    written = build()
    for t, n in written.items():
        print(f"  {t}: {n} rows")
    print(f"Wrote {len(written)} tables -> {DB_PATH}")
    if not written:
        print("No result CSVs found. Run predict.py / stock.py / inventory_analytics.py first.")


if __name__ == "__main__":
    main()
