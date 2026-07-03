"""Tests for the SQLite data layer (read from DB, fall back to CSV)."""

import sqlite3

import pandas as pd

from build_db import build
from src.datastore import load


def _make_csvs(results_dir):
    (results_dir / "predictions.csv").write_text(
        "Date,Store ID,Product ID,Predicted_Demand\n2024-01-01,S001,P0001,100\n"
    )
    (results_dir / "stock_recommendations.csv").write_text(
        "Store ID,Product ID,avg_daily_demand\nS001,P0001,92.5\n"
    )


def test_build_creates_tables(tmp_path):
    results = tmp_path / "results"; results.mkdir()
    _make_csvs(results)
    db = tmp_path / "analysis.db"
    written = build(results=results, db_path=db)
    assert written == {"predictions": 1, "stock_recommendations": 1}
    con = sqlite3.connect(db)
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    con.close()
    assert {"predictions", "stock_recommendations"} <= tables


def test_load_prefers_db(tmp_path):
    results = tmp_path / "results"; results.mkdir()
    _make_csvs(results)
    db = tmp_path / "analysis.db"
    build(results=results, db_path=db)
    # Change the CSV after building the DB; load() should return the DB value.
    (results / "predictions.csv").write_text("Date,Store ID,Product ID,Predicted_Demand\n2024-01-01,S999,P9999,7\n")
    df = load("predictions", results / "predictions.csv", db_path=db)
    assert df.iloc[0]["Store ID"] == "S001"   # from the DB, not the edited CSV


def test_load_falls_back_to_csv_when_no_db(tmp_path):
    results = tmp_path / "results"; results.mkdir()
    _make_csvs(results)
    df = load("predictions", results / "predictions.csv", db_path=tmp_path / "missing.db")
    assert len(df) == 1 and df.iloc[0]["Store ID"] == "S001"


def test_load_falls_back_when_table_missing(tmp_path):
    results = tmp_path / "results"; results.mkdir()
    _make_csvs(results)
    db = tmp_path / "analysis.db"
    build(results=results, db_path=db)          # has predictions, not "anomalies"
    (results / "anomalies.csv").write_text("Store ID,anomaly_type\nS001,spike\n")
    df = load("anomalies", results / "anomalies.csv", db_path=db)
    assert df.iloc[0]["anomaly_type"] == "spike"   # read from CSV fallback
