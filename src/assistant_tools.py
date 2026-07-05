"""
Tools for the Demand & Stock assistant.

These are plain Python functions over the model outputs (forecasts and stock
recommendations). They are deliberately LLM-agnostic and self-contained so they
can be:
  * unit-tested / run directly without any API key, and
  * exposed to Azure OpenAI as callable tools (see assistant.py).

Each tool returns JSON-serializable dicts/lists. TOOL_SPECS holds the matching
OpenAI function-calling schema, and dispatch() routes a tool name + args to the
right function.
"""

import json
from pathlib import Path

import pandas as pd

from src.datastore import load as _load_table

PRED_PATH = Path("results/predictions.csv")
STOCK_PATH = Path("results/stock_recommendations.csv")
INV_PATH = Path("results/inventory_analytics.csv")
ADV_PATH = Path("results/advanced_analytics.csv")
ANOM_PATH = Path("results/anomalies.csv")
OPT_PATH = Path("results/optimization_allocation.csv")
OPT_SUMMARY_PATH = Path("results/optimization_summary.json")
FIN_PATH = Path("results/financial_metrics.csv")
FIN_SUMMARY_PATH = Path("results/financial_summary.json")

_pred_cache = None
_stock_cache = None
_inv_cache = None
_adv_cache = None
_anom_cache = None
_opt_cache = None
_fin_cache = None


# Loaders read from db/analysis.db when present, otherwise the result CSV.
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


def _financials() -> pd.DataFrame:
    global _fin_cache
    if _fin_cache is None:
        try:
            _fin_cache = _load_table("financial_metrics", FIN_PATH)
        except Exception:
            _fin_cache = pd.DataFrame()
    return _fin_cache


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
def list_series() -> dict:
    """Return the available store and product IDs."""
    s = _stock()
    return {
        "stores": sorted(s["Store ID"].unique().tolist()),
        "products": sorted(s["Product ID"].unique().tolist()),
    }


def get_demand_forecast(store_id: str, product_id: str, last_n_days: int = 7) -> dict:
    """Recent forecasted vs actual demand for one store-product series."""
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
    """Inventory policy (safety stock, service level, savings) for one series."""
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
    """Series with the lowest achieved service level (highest stock-out risk)."""
    s = _stock().sort_values("service_model").head(top_n)
    return {
        "top_risks": s[
            ["Store ID", "Product ID", "service_model", "demand_std", "safety_stock_model"]
        ].to_dict("records")
    }


def inventory_summary() -> dict:
    """Portfolio-wide inventory comparison: naive vs model-based policy."""
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


# --------------------------------------------------------------------------- #
# OpenAI function-calling schema + dispatch
# --------------------------------------------------------------------------- #
def abc_summary() -> dict:
    """ABC classification breakdown: how many SKUs and what revenue share per class."""
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
    """EOQ, reorder point, ABC class and stock alert for one store-product."""
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
    """SKUs needing action, most valuable first. status: CRITICAL, REORDER, or all."""
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
    """ABC-XYZ class, newsvendor order qty, z-score safety stock, turnover for a SKU."""
    adv = _advanced()
    m = adv[(adv["Store ID"] == store_id) & (adv["Product ID"] == product_id)]
    if m.empty:
        return {"error": f"No advanced policy for {store_id}/{product_id}."}
    r = m.iloc[0]
    return {
        "store_id": store_id,
        "product_id": product_id,
        "abc_xyz": r["abc_xyz"],
        "demand_cv": float(r["cv"]),
        "newsvendor_order_qty": float(r["newsvendor_qty"]),
        "safety_stock_zscore": float(r["safety_stock_zscore"]),
        "critical_ratio": float(r["critical_ratio"]),
        "turnover_per_year": float(r["turnover"]),
        "days_of_stock": float(r["days_of_stock"]),
    }


def list_anomalies(top_n: int = 10) -> dict:
    """Detected demand/stock anomalies with a plain-language reason."""
    anom = _anomalies()
    if anom.empty:
        return {"count": 0, "anomalies": []}
    return {"count": int(len(anom)), "anomalies": anom.head(top_n).to_dict("records")}


PRICE_ELASTICITY = -1.2  # assumed retail price elasticity for scenario planning


def whatif_simulasyon(store_id: str, product_id: str, price_change_pct: float) -> dict:
    """What-if: demand and revenue impact of a price change (elasticity -1.2)."""
    inv = _inventory()
    m = inv[(inv["Store ID"] == store_id) & (inv["Product ID"] == product_id)]
    if m.empty:
        return {"error": f"No data for {store_id}/{product_id}."}
    r = m.iloc[0]
    price, demand = float(r["unit_price"]), float(r["avg_daily_demand"])
    f = 1 + price_change_pct / 100.0
    if f <= 0:
        return {"error": "Price change too negative."}
    demand_mult = f ** PRICE_ELASTICITY
    new_price, new_demand = price * f, demand * demand_mult
    cur_rev, new_rev = price * demand, new_price * new_demand
    return {
        "store_id": store_id, "product_id": product_id,
        "price_change_pct": price_change_pct, "elasticity_assumed": PRICE_ELASTICITY,
        "current": {"price": round(price, 2), "daily_demand": round(demand, 1),
                    "daily_revenue": round(cur_rev, 0)},
        "scenario": {"price": round(new_price, 2), "daily_demand": round(new_demand, 1),
                     "daily_revenue": round(new_rev, 0)},
        "demand_change_pct": round((demand_mult - 1) * 100, 1),
        "revenue_change_pct": round((new_rev / cur_rev - 1) * 100, 1),
    }


def pdf_rapor() -> dict:
    """Download link for the default product/stock PDF report."""
    return {
        "download_url": "/report.pdf",
        "filename": "talep_stok_raporu.pdf",
        "note": "Varsayılan ürün/stok raporu hazır — linke tıklayıp indirin.",
    }


RAG_MIN_SCORE = 0.30  # below this the query is treated as out-of-knowledge-base


def bilgi_ara(query: str, top_k: int = 3) -> dict:
    """RAG: retrieve knowledge-base chunks for a conceptual question.

    Returns sources+chunks to ground the answer, or found=False when nothing in
    the knowledge base is relevant (the assistant should then say it doesn't know).
    """
    try:
        from src.rag import search

        hits = search(query, top_k=top_k)
        if not hits or hits[0]["score"] < RAG_MIN_SCORE:
            return {"found": False, "note": "Bilgi tabanında bu konuyla ilgili içerik yok."}
        return {"found": True, "sources": hits}
    except Exception as e:
        return {"error": str(e)}


def finansal_ozet(top_n: int = 5) -> dict:
    """Financial-metric summary in money terms: revenue, gross profit/margin, the
    ₺ saved by inventory reduction, portfolio turnover, and promotion ROI. Also
    returns the most profitable SKUs. Numbers use the assumptions in the summary.
    """
    fin = _financials()
    if fin.empty:
        return {"error": "Finansal metrikler bulunamadı. Önce financial_metrics.py çalıştırın."}
    try:
        with open(FIN_SUMMARY_PATH) as f:
            summary = json.load(f)
    except Exception:
        summary = {}
    cols = ["Store ID", "Product ID", "abc_class", "revenue", "gross_profit",
            "turnover", "promo_roi_pct"]
    top = fin.sort_values("gross_profit", ascending=False).head(top_n)
    return {"summary": summary, "top_by_profit": top[cols].to_dict("records")}


def optimizasyon_onerisi(top_n: int = 8) -> dict:
    """Multi-store stock-allocation optimisation result (PuLP linear program).

    A budget/capacity/service-constrained LP decides how many units to allocate
    to each store-product to minimise total holding + stockout cost. Returns the
    summary and the biggest recommended replenishment orders.
    """
    alloc = _optimization()
    if alloc.empty:
        return {"error": "Optimizasyon sonucu bulunamadı. Önce optimize_allocation.py çalıştırın."}
    try:
        with open(OPT_SUMMARY_PATH) as f:
            summary = json.load(f)
    except Exception:
        summary = {}
    top = alloc.sort_values("recommended_order", ascending=False).head(top_n)
    cols = ["Store ID", "Product ID", "abc_class", "service_target",
            "allocation", "current_inventory", "recommended_order", "service_fill_pct"]
    return {"summary": summary, "top_orders": top[cols].to_dict("records")}


def yonetici_ozeti() -> dict:
    """Full data snapshot for an executive summary: KPIs, ABC-XYZ, alerts, anomalies.

    The assistant should turn this into a decision-focused Turkish summary with
    product-level commentary and an explanation of the anomalies.
    """
    inv = _inventory()
    stock = _stock()
    adv = _advanced()
    naive = float(stock["avg_inventory_naive"].sum())
    model = float(stock["avg_inventory_model"].sum())

    top_alerts = (
        inv[inv["alert_status"] != "OK"].sort_values("annual_revenue", ascending=False).head(5)
    )
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
    {
        "type": "function",
        "function": {
            "name": "list_series",
            "description": "List available store IDs and product IDs.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_demand_forecast",
            "description": "Recent forecasted vs actual daily demand for a store-product.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_id": {"type": "string", "description": "e.g. S001"},
                    "product_id": {"type": "string", "description": "e.g. P0001"},
                    "last_n_days": {"type": "integer", "default": 7},
                },
                "required": ["store_id", "product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_recommendation",
            "description": "Safety stock, service level, and inventory savings for a store-product.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_id": {"type": "string"},
                    "product_id": {"type": "string"},
                },
                "required": ["store_id", "product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_top_stockout_risks",
            "description": "Store-products with the lowest achieved service level (highest risk).",
            "parameters": {
                "type": "object",
                "properties": {"top_n": {"type": "integer", "default": 5}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inventory_summary",
            "description": "Portfolio-wide inventory: naive vs model policy and total savings.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "abc_summary",
            "description": "ABC analysis: SKU count and revenue share for classes A, B, C.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_inventory_policy",
            "description": "EOQ (order quantity), reorder point, ABC class, current stock and alert for a store-product.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_id": {"type": "string"},
                    "product_id": {"type": "string"},
                },
                "required": ["store_id", "product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_stockout_alerts",
            "description": "SKUs needing action (most valuable first). Optionally filter by status CRITICAL or REORDER.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["CRITICAL", "REORDER"]},
                    "top_n": {"type": "integer", "default": 10},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_advanced_policy",
            "description": "ABC-XYZ segment, newsvendor order quantity, z-score safety stock, and stock turnover for a store-product.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_id": {"type": "string"},
                    "product_id": {"type": "string"},
                },
                "required": ["store_id", "product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_anomalies",
            "description": "Detected demand/stock anomalies (spikes, drops, critical or excess stock) with reasons.",
            "parameters": {"type": "object", "properties": {"top_n": {"type": "integer", "default": 10}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "yonetici_ozeti",
            "description": "Full snapshot (KPIs, ABC/ABC-XYZ, top alerts, anomalies) to build an executive summary. Call this when the user asks for a yönetici özeti / executive summary.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finansal_ozet",
            "description": "Financial summary in money terms: ₺ revenue (demand×price), gross profit and margin, the ₺ saved by the model's inventory reduction (freed capital + annual holding-cost savings), portfolio inventory turnover, and promotion ROI with average demand lift. Call for 'finansal özet', 'kâr/marj', 'ciro', 'promosyon ROI', 'stok devir hızı'. Returns the summary plus the most profitable SKUs; numbers use the gross-margin and holding-cost assumptions stated in the result.",
            "parameters": {"type": "object", "properties": {"top_n": {"type": "integer", "default": 5}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "optimizasyon_onerisi",
            "description": "Multi-store stock-ALLOCATION optimisation result from a real linear program (PuLP): under a procurement budget, warehouse-capacity limit and minimum service level, it decides how many units to allocate to each store-product to minimise total holding + stockout cost. Call for 'optimizasyon önerisi', 'stok tahsisi', 'bütçeyle en iyi dağıtım', and ALSO for feasibility/sensitivity ('duyarlılık analizi', 'gölge fiyat', 'ne zaman infeasible olur'). Returns the summary — budget/capacity used, total cost, service level, savings vs an even-cut baseline, budget/capacity shadow prices, the min feasible budget/capacity, and a budget-sweep sensitivity curve — plus the biggest recommended orders.",
            "parameters": {"type": "object", "properties": {"top_n": {"type": "integer", "default": 8}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bilgi_ara",
            "description": "Knowledge-base search (RAG) for CONCEPTUAL questions — what/why/how definitions of methods like ABC, ABC-XYZ, EOQ, newsvendor, safety stock, reorder point, quantile forecasting, turnover, and the project methodology. Use for explanations, NOT for live numbers (use the data tools for those).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "the conceptual question"},
                    "top_k": {"type": "integer", "default": 3},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "whatif_simulasyon",
            "description": "What-if price simulation for a store-product: demand and revenue impact of a percentage price change (e.g. 'fiyatı %10 artırırsam ne olur'). price_change_pct is the percent change (10 for +10%).",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_id": {"type": "string"},
                    "product_id": {"type": "string"},
                    "price_change_pct": {"type": "number"},
                },
                "required": ["store_id", "product_id", "price_change_pct"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pdf_rapor",
            "description": "Give the download link for the default product/stock PDF report. Call when the user asks to create/download a PDF or report; do not ask for a filename or logo.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def dispatch(name: str, arguments: dict) -> dict:
    """Route a tool call to its function; return a JSON-serializable result."""
    if name not in _FUNCS:
        return {"error": f"Unknown tool: {name}"}
    try:
        return _FUNCS[name](**(arguments or {}))
    except Exception as e:  # surface errors back to the model, don't crash
        return {"error": str(e)}


if __name__ == "__main__":
    # Smoke test the tools without any LLM / API key.
    print("list_series:", json.dumps(list_series(), indent=2)[:200], "...\n")
    print("inventory_summary:", json.dumps(inventory_summary(), indent=2), "\n")
    print("top risks:", json.dumps(list_top_stockout_risks(3), indent=2), "\n")
    ex = list_series()
    st, pr = ex["stores"][0], ex["products"][0]
    print(f"forecast {st}/{pr}:", json.dumps(get_demand_forecast(st, pr, 3), indent=2), "\n")
    print(f"stock {st}/{pr}:", json.dumps(get_stock_recommendation(st, pr), indent=2))
