"""Assistant tools for the web backend.

Same logic as src/assistant_tools.py, but the CSV paths are resolved relative to
this file (so it works both locally and inside the container). Reads the
forecast + stock-recommendation outputs and exposes them as callable tools.
"""

import json
import os
from pathlib import Path

import pandas as pd

from store import load as _load_table

DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).parent / "data"))
PRED_PATH = DATA_DIR / "predictions.csv"
STOCK_PATH = DATA_DIR / "stock_recommendations.csv"
INV_PATH = DATA_DIR / "inventory_analytics.csv"
ADV_PATH = DATA_DIR / "advanced_analytics.csv"
ANOM_PATH = DATA_DIR / "anomalies.csv"

_pred_cache = None
_stock_cache = None
_inv_cache = None
_adv_cache = None
_anom_cache = None


# Loaders read from analysis.db when present, otherwise the bundled CSV.
def _predictions() -> pd.DataFrame:
    global _pred_cache
    if _pred_cache is None:
        _pred_cache = _load_table("predictions", PRED_PATH)
    return _pred_cache


def _stock() -> pd.DataFrame:
    global _stock_cache
    if _stock_cache is None:
        _stock_cache = _load_table("stock_recommendations", STOCK_PATH)
    return _stock_cache


def _inventory() -> pd.DataFrame:
    global _inv_cache
    if _inv_cache is None:
        _inv_cache = _load_table("inventory_analytics", INV_PATH)
    return _inv_cache


def _advanced() -> pd.DataFrame:
    global _adv_cache
    if _adv_cache is None:
        _adv_cache = _load_table("advanced_analytics", ADV_PATH)
    return _adv_cache


def _anomalies() -> pd.DataFrame:
    global _anom_cache
    if _anom_cache is None:
        try:
            _anom_cache = _load_table("anomalies", ANOM_PATH)
        except Exception:
            _anom_cache = pd.DataFrame()
    return _anom_cache


def list_series() -> dict:
    s = _stock()
    return {
        "stores": sorted(s["Store ID"].unique().tolist()),
        "products": sorted(s["Product ID"].unique().tolist()),
    }


def get_demand_forecast(store_id: str, product_id: str, last_n_days: int = 7) -> dict:
    df = _predictions()
    m = df[(df["Store ID"] == store_id) & (df["Product ID"] == product_id)]
    if m.empty:
        return {"error": f"No data for {store_id}/{product_id}."}
    m = m.sort_values("Date").tail(last_n_days)
    return {
        "store_id": store_id,
        "product_id": product_id,
        "rows": m[["Date", "Predicted_Demand", "Actual_Demand"]].to_dict("records"),
        "avg_predicted": round(float(m["Predicted_Demand"].mean()), 1),
    }


def get_stock_recommendation(store_id: str, product_id: str) -> dict:
    s = _stock()
    m = s[(s["Store ID"] == store_id) & (s["Product ID"] == product_id)]
    if m.empty:
        return {"error": f"No recommendation for {store_id}/{product_id}."}
    r = m.iloc[0]
    return {
        "store_id": store_id,
        "product_id": product_id,
        "avg_daily_demand": float(r["avg_daily_demand"]),
        "safety_stock_recommended": float(r["safety_stock_model"]),
        "safety_stock_naive": float(r["safety_stock_naive"]),
        "avg_inventory_recommended": float(r["avg_inventory_model"]),
        "service_level_achieved": float(r["service_model"]),
        "inventory_reduction_pct": float(r["inventory_reduction_%"]),
    }


def list_top_stockout_risks(top_n: int = 5) -> dict:
    s = _stock().sort_values("service_model").head(top_n)
    return {
        "top_risks": s[
            ["Store ID", "Product ID", "service_model", "demand_std", "safety_stock_model"]
        ].to_dict("records")
    }


def inventory_summary() -> dict:
    s = _stock()
    naive = float(s["avg_inventory_naive"].sum())
    model = float(s["avg_inventory_model"].sum())
    return {
        "series_count": int(len(s)),
        "total_inventory_naive": round(naive),
        "total_inventory_model": round(model),
        "inventory_reduction_pct": round((naive - model) / naive * 100, 1),
        "avg_service_level_model": round(float(s["service_model"].mean()), 3),
    }


def abc_summary() -> dict:
    inv = _inventory()
    total = inv["annual_revenue"].sum()
    out = []
    for cls in ["A", "B", "C"]:
        sub = inv[inv["abc_class"] == cls]
        out.append({
            "class": cls,
            "sku_count": int(len(sub)),
            "revenue_share_pct": round(float(sub["annual_revenue"].sum() / total * 100), 1),
        })
    return {"abc_breakdown": out}


def get_inventory_policy(store_id: str, product_id: str) -> dict:
    inv = _inventory()
    m = inv[(inv["Store ID"] == store_id) & (inv["Product ID"] == product_id)]
    if m.empty:
        return {"error": f"No inventory policy for {store_id}/{product_id}."}
    r = m.iloc[0]
    return {
        "store_id": store_id,
        "product_id": product_id,
        "abc_class": r["abc_class"],
        "economic_order_quantity": float(r["EOQ"]),
        "reorder_point": float(r["reorder_point"]),
        "safety_stock": float(r["safety_stock_model"]),
        "current_inventory": float(r["current_inventory"]),
        "days_of_cover": float(r["days_of_cover"]),
        "alert_status": r["alert_status"],
    }


def list_stockout_alerts(status: str = None, top_n: int = 10) -> dict:
    inv = _inventory()
    alerts = inv[inv["alert_status"] != "OK"]
    if status:
        alerts = alerts[alerts["alert_status"] == status.upper()]
    total = int(len(alerts))  # full count (matches the dashboard banner)
    view = alerts.sort_values("annual_revenue", ascending=False).head(top_n)[
        ["Store ID", "Product ID", "alert_status", "abc_class",
         "current_inventory", "reorder_point", "EOQ"]
    ].rename(columns={"EOQ": "economic_order_quantity"})
    return {"total": total, "count": int(len(view)), "alerts": view.to_dict("records")}


def get_advanced_policy(store_id: str, product_id: str) -> dict:
    adv = _advanced()
    m = adv[(adv["Store ID"] == store_id) & (adv["Product ID"] == product_id)]
    if m.empty:
        return {"error": f"No advanced policy for {store_id}/{product_id}."}
    r = m.iloc[0]
    return {
        "store_id": store_id, "product_id": product_id, "abc_xyz": r["abc_xyz"],
        "demand_cv": float(r["cv"]), "newsvendor_order_qty": float(r["newsvendor_qty"]),
        "safety_stock_zscore": float(r["safety_stock_zscore"]),
        "critical_ratio": float(r["critical_ratio"]),
        "turnover_per_year": float(r["turnover"]), "days_of_stock": float(r["days_of_stock"]),
    }


def list_anomalies(top_n: int = 10) -> dict:
    anom = _anomalies()
    if anom.empty:
        return {"count": 0, "anomalies": []}
    return {"count": int(len(anom)), "anomalies": anom.head(top_n).to_dict("records")}


PRICE_ELASTICITY = -1.2  # assumed retail price elasticity for scenario planning


def whatif_simulasyon(store_id: str, product_id: str, price_change_pct: float) -> dict:
    """What-if: estimate demand and revenue impact of a price change for a SKU.

    Uses an assumed price elasticity (-1.2): demand scales as (1 + Δ)^elasticity.
    """
    inv = _inventory()
    m = inv[(inv["Store ID"] == store_id) & (inv["Product ID"] == product_id)]
    if m.empty:
        return {"error": f"No data for {store_id}/{product_id}."}
    r = m.iloc[0]
    price = float(r["unit_price"])
    demand = float(r["avg_daily_demand"])
    f = 1 + price_change_pct / 100.0
    if f <= 0:
        return {"error": "Price change too negative."}
    demand_mult = f ** PRICE_ELASTICITY
    new_price, new_demand = price * f, demand * demand_mult
    cur_rev, new_rev = price * demand, new_price * new_demand
    return {
        "store_id": store_id, "product_id": product_id,
        "price_change_pct": price_change_pct,
        "elasticity_assumed": PRICE_ELASTICITY,
        "current": {"price": round(price, 2), "daily_demand": round(demand, 1),
                    "daily_revenue": round(cur_rev, 0)},
        "scenario": {"price": round(new_price, 2), "daily_demand": round(new_demand, 1),
                     "daily_revenue": round(new_rev, 0)},
        "demand_change_pct": round((demand_mult - 1) * 100, 1),
        "revenue_change_pct": round((new_rev / cur_rev - 1) * 100, 1),
    }


RAG_MIN_SCORE = 0.30


def bilgi_ara(query: str, top_k: int = 3) -> dict:
    """RAG knowledge-base search for conceptual questions (with relevance floor)."""
    try:
        from rag import search

        hits = search(query, top_k=top_k)
        if not hits or hits[0]["score"] < RAG_MIN_SCORE:
            return {"found": False, "note": "Bilgi tabanında bu konuyla ilgili içerik yok."}
        return {"found": True, "sources": hits}
    except Exception as e:
        return {"error": str(e)}


def yonetici_ozeti() -> dict:
    inv = _inventory()
    stock = _stock()
    adv = _advanced()
    naive = float(stock["avg_inventory_naive"].sum())
    model = float(stock["avg_inventory_model"].sum())
    top_alerts = inv[inv["alert_status"] != "OK"].sort_values("annual_revenue", ascending=False).head(5)
    return {
        "kpis": {
            "total_skus": int(len(inv)),
            "inventory_reduction_pct": round((naive - model) / naive * 100, 1),
            "avg_service_level_pct": round(float(stock["service_model"].mean()) * 100, 1),
            "reorder_alerts": int((inv["alert_status"] != "OK").sum()),
            "avg_turnover": round(float(adv["turnover"].mean()), 1),
        },
        "abc_breakdown": abc_summary()["abc_breakdown"],
        "abc_xyz_counts": adv["abc_xyz"].value_counts().to_dict(),
        "top_value_alerts": top_alerts[
            ["Store ID", "Product ID", "abc_class", "alert_status", "current_inventory", "reorder_point"]
        ].to_dict("records"),
        "anomalies": _anomalies().to_dict("records") if not _anomalies().empty else [],
    }


_FUNCS = {
    "list_series": list_series,
    "get_demand_forecast": get_demand_forecast,
    "get_stock_recommendation": get_stock_recommendation,
    "list_top_stockout_risks": list_top_stockout_risks,
    "inventory_summary": inventory_summary,
    "abc_summary": abc_summary,
    "get_inventory_policy": get_inventory_policy,
    "list_stockout_alerts": list_stockout_alerts,
    "get_advanced_policy": get_advanced_policy,
    "list_anomalies": list_anomalies,
    "yonetici_ozeti": yonetici_ozeti,
    "bilgi_ara": bilgi_ara,
    "whatif_simulasyon": whatif_simulasyon,
}

TOOL_SPECS = [
    {"type": "function", "function": {"name": "list_series",
        "description": "List available store IDs and product IDs.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "get_demand_forecast",
        "description": "Recent forecasted vs actual daily demand for a store-product.",
        "parameters": {"type": "object", "properties": {
            "store_id": {"type": "string", "description": "e.g. S001"},
            "product_id": {"type": "string", "description": "e.g. P0001"},
            "last_n_days": {"type": "integer", "default": 7}},
            "required": ["store_id", "product_id"]}}},
    {"type": "function", "function": {"name": "get_stock_recommendation",
        "description": "Safety stock, service level, and inventory savings for a store-product.",
        "parameters": {"type": "object", "properties": {
            "store_id": {"type": "string"}, "product_id": {"type": "string"}},
            "required": ["store_id", "product_id"]}}},
    {"type": "function", "function": {"name": "list_top_stockout_risks",
        "description": "Store-products with the lowest achieved service level (highest risk).",
        "parameters": {"type": "object", "properties": {"top_n": {"type": "integer", "default": 5}}}}},
    {"type": "function", "function": {"name": "inventory_summary",
        "description": "Portfolio-wide inventory: naive vs model policy and total savings.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "abc_summary",
        "description": "ABC analysis: SKU count and revenue share for classes A, B, C.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "get_inventory_policy",
        "description": "EOQ (order quantity), reorder point, ABC class, current stock and alert for a store-product.",
        "parameters": {"type": "object", "properties": {
            "store_id": {"type": "string"}, "product_id": {"type": "string"}},
            "required": ["store_id", "product_id"]}}},
    {"type": "function", "function": {"name": "list_stockout_alerts",
        "description": "SKUs needing action (most valuable first). Optionally filter by status CRITICAL or REORDER.",
        "parameters": {"type": "object", "properties": {
            "status": {"type": "string", "enum": ["CRITICAL", "REORDER"]},
            "top_n": {"type": "integer", "default": 10}}}}},
    {"type": "function", "function": {"name": "get_advanced_policy",
        "description": "ABC-XYZ segment, newsvendor order quantity, z-score safety stock, and stock turnover for a store-product.",
        "parameters": {"type": "object", "properties": {
            "store_id": {"type": "string"}, "product_id": {"type": "string"}},
            "required": ["store_id", "product_id"]}}},
    {"type": "function", "function": {"name": "list_anomalies",
        "description": "Detected demand/stock anomalies (spikes, drops, critical or excess stock) with reasons.",
        "parameters": {"type": "object", "properties": {"top_n": {"type": "integer", "default": 10}}}}},
    {"type": "function", "function": {"name": "yonetici_ozeti",
        "description": "Full snapshot (KPIs, ABC/ABC-XYZ, top alerts, anomalies) to build an executive summary. Call when the user asks for a yönetici özeti / executive summary.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "bilgi_ara",
        "description": "Knowledge-base search (RAG) for CONCEPTUAL questions — definitions/why/how of ABC, ABC-XYZ, EOQ, newsvendor, safety stock, reorder point, quantile forecasting, turnover, and the project methodology. Use for explanations, NOT for live numbers.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "top_k": {"type": "integer", "default": 3}},
            "required": ["query"]}}},
    {"type": "function", "function": {"name": "whatif_simulasyon",
        "description": "What-if price simulation for a store-product: estimates the demand and revenue impact of a percentage price change (e.g. 'fiyatı %10 artırırsam ne olur'). price_change_pct is the percent change (10 for +10%, -5 for -5%).",
        "parameters": {"type": "object", "properties": {
            "store_id": {"type": "string"}, "product_id": {"type": "string"},
            "price_change_pct": {"type": "number"}},
            "required": ["store_id", "product_id", "price_change_pct"]}}},
]


def dispatch(name: str, arguments: dict) -> dict:
    if name not in _FUNCS:
        return {"error": f"Unknown tool: {name}"}
    try:
        return _FUNCS[name](**(arguments or {}))
    except Exception as e:
        return {"error": str(e)}
