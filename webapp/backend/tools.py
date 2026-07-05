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
OPT_PATH = DATA_DIR / "optimization_allocation.csv"
OPT_SUMMARY_PATH = DATA_DIR / "optimization_summary.json"
FIN_PATH = DATA_DIR / "financial_metrics.csv"
FIN_SUMMARY_PATH = DATA_DIR / "financial_summary.json"

_pred_cache = None
_stock_cache = None
_inv_cache = None
_adv_cache = None
_anom_cache = None
_opt_cache = None
_fin_cache = None


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


def _optimization() -> pd.DataFrame:
    global _opt_cache
    if _opt_cache is None:
        try:
            _opt_cache = _load_table("optimization_allocation", OPT_PATH)
        except Exception:
            _opt_cache = pd.DataFrame()
    return _opt_cache


def _opt_summary() -> dict:
    try:
        with open(OPT_SUMMARY_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _financials() -> pd.DataFrame:
    global _fin_cache
    if _fin_cache is None:
        try:
            _fin_cache = _load_table("financial_metrics", FIN_PATH)
        except Exception:
            _fin_cache = pd.DataFrame()
    return _fin_cache


def _fin_summary() -> dict:
    try:
        with open(FIN_SUMMARY_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


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


def pdf_rapor() -> dict:
    """Return the download link for the default product/stock PDF report."""
    return {
        "download_url": "/report.pdf",
        "filename": "talep_stok_raporu.pdf",
        "note": "Varsayılan ürün/stok raporu hazır — linke tıklayıp indirin.",
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


def finansal_ozet(top_n: int = 5) -> dict:
    """Financial-metric summary: ₺ revenue, gross profit/margin, the money saved
    by the model's inventory reduction (freed capital + annual holding savings),
    portfolio inventory turnover, and promotion ROI. Also returns the most
    profitable SKUs. Numbers assume a gross-margin and holding-cost rate stated
    in the summary."""
    fin = _financials()
    summary = _fin_summary()
    if fin.empty:
        return {"error": "Finansal metrikler bulunamadı. Önce financial_metrics.py çalıştırın."}
    cols = ["Store ID", "Product ID", "abc_class", "revenue", "gross_profit",
            "turnover", "promo_roi_pct"]
    top = fin.sort_values("gross_profit", ascending=False).head(top_n)
    return {"summary": summary, "top_by_profit": top[cols].to_dict("records")}


def optimizasyon_onerisi(top_n: int = 8) -> dict:
    """Multi-store stock-allocation optimisation result. A PuLP linear program
    minimises total holding + stockout cost under a procurement budget, a
    warehouse-capacity limit and a minimum service level, deciding how many units
    to allocate to each store-product. Returns the summary and the biggest
    recommended replenishment orders."""
    alloc = _optimization()
    if alloc.empty:
        return {"error": "Optimizasyon sonucu bulunamadı. Önce optimize_allocation.py çalıştırın."}
    top = alloc.sort_values("recommended_order", ascending=False).head(top_n)
    cols = ["Store ID", "Product ID", "abc_class", "service_target",
            "allocation", "current_inventory", "recommended_order", "service_fill_pct"]
    return {"summary": _opt_summary(), "top_orders": top[cols].to_dict("records")}


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
    "optimizasyon_onerisi": optimizasyon_onerisi,
    "finansal_ozet": finansal_ozet,
    "bilgi_ara": bilgi_ara,
    "whatif_simulasyon": whatif_simulasyon,
    "pdf_rapor": pdf_rapor,
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
    {"type": "function", "function": {"name": "finansal_ozet",
        "description": "Financial summary in money terms: ₺ revenue (demand×price), gross profit and margin, the ₺ saved by the model's inventory reduction (freed capital + annual holding-cost savings), portfolio inventory turnover, and promotion ROI with the average demand lift. Call for 'finansal özet', 'kâr/marj', 'ciro', 'stok azaltımı ne kadar tasarruf', 'promosyon ROI', 'stok devir hızı'. Returns the summary plus the most profitable SKUs. Numbers use the gross-margin and holding-cost assumptions stated in the result.",
        "parameters": {"type": "object", "properties": {"top_n": {"type": "integer", "default": 5}}}}},
    {"type": "function", "function": {"name": "optimizasyon_onerisi",
        "description": "Multi-store stock-ALLOCATION optimisation result from a real linear program (PuLP): under a procurement budget, warehouse-capacity limit and minimum service level, it decides how many units to allocate to each store-product to minimise total holding + stockout cost. Call for 'optimizasyon önerisi', 'stok tahsisi', 'bütçeyle en iyi dağıtım', 'hangi ürüne ne kadar sipariş', and ALSO for feasibility/sensitivity questions ('duyarlılık analizi', 'gölge fiyat', 'bütçe/kapasite değişirse', 'ne zaman infeasible olur'). Returns the summary — budget/capacity used, total cost, service level, savings vs an even-cut baseline, the budget/capacity SHADOW PRICES (₺ cost saved per one extra unit of the resource; 0 means that constraint is slack), the MIN FEASIBLE budget/capacity below which no plan exists, and a budget-sweep sensitivity curve — plus the biggest recommended orders.",
        "parameters": {"type": "object", "properties": {"top_n": {"type": "integer", "default": 8}}}}},
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
    {"type": "function", "function": {"name": "pdf_rapor",
        "description": "Give the download link for the default product/stock PDF report. Call this when the user asks to create/download a PDF or report; do not ask for a filename or logo.",
        "parameters": {"type": "object", "properties": {}}}},
]


def dispatch(name: str, arguments: dict) -> dict:
    if name not in _FUNCS:
        return {"error": f"Unknown tool: {name}"}
    try:
        return _FUNCS[name](**(arguments or {}))
    except Exception as e:
        return {"error": str(e)}
