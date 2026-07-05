"""
Financial-metric layer on top of the demand + inventory analytics.

Turns the operational outputs (units, service levels, safety stock) into the
numbers a finance team actually reports on:

  1. Inventory-reduction savings — the model holds less stock than a naive
     policy; value the freed capital and the annual holding cost saved (₺).
  2. Revenue            — annual demand × price.
  3. Gross profit / margin — revenue × assumed gross-margin rate.
  4. Inventory turnover — annual demand ÷ average inventory.
  5. Promotion ROI      — net incremental gross profit from promo-driven sales
     divided by the discount cost of running the promotions.

Two assumptions have no source in the data and are explicit, tunable inputs:
  * gross_margin_rate — gross margin as a fraction of revenue (default 0.35).
  * holding_rate      — annual holding cost as a fraction of unit price (0.25).

Inputs:
  results/inventory_analytics.csv     price, annual revenue, avg daily demand
  results/stock_recommendations.csv   naive vs model average inventory
  data/demand_forecasting.csv         promotion flag, discount, realised demand

Outputs:
  results/financial_metrics.csv       per-SKU financial table
  results/financial_summary.json      portfolio totals + assumptions
  results/19_financial_summary.png    gross profit by ABC class

Run:
  .venv/bin/python financial_metrics.py
  .venv/bin/python financial_metrics.py --gross-margin 0.4 --holding-rate 0.2
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path("data/demand_forecasting.csv")
INV_PATH = Path("results/inventory_analytics.csv")
STOCK_PATH = Path("results/stock_recommendations.csv")
RESULTS_DIR = Path("results")
DAYS_PER_YEAR = 365


def promotion_roi(raw, gross_margin_rate):
    """Portfolio + per-SKU promotion ROI from the raw transaction panel.

    For each store-product, compare demand on promotion days with its own
    non-promotion baseline. The *incremental* units the promo drove earn gross
    profit; the discount handed out on every promo-day unit is the cost.

        ROI = (incremental gross profit − discount cost) / discount cost
    """
    keys = ["Store ID", "Product ID"]
    df = raw.copy()
    df["disc"] = df["Discount"] / 100.0

    base = (df[df["Promotion"] == 0].groupby(keys)["Demand"].mean()
            .rename("base_demand").reset_index())
    promo = df[df["Promotion"] == 1].merge(base, on=keys, how="left")
    promo["base_demand"] = promo["base_demand"].fillna(promo["Demand"])
    promo["inc_units"] = (promo["Demand"] - promo["base_demand"]).clip(lower=0)
    promo["inc_profit"] = promo["inc_units"] * promo["Price"] * gross_margin_rate
    promo["promo_cost"] = promo["Demand"] * promo["Price"] * promo["disc"]

    per_sku = promo.groupby(keys).agg(
        promo_days=("Demand", "size"),
        promo_demand=("Demand", "mean"),
        base_demand=("base_demand", "mean"),
        inc_profit=("inc_profit", "sum"),
        promo_cost=("promo_cost", "sum"),
    ).reset_index()
    per_sku["promo_lift_pct"] = np.where(
        per_sku["base_demand"] > 0,
        (per_sku["promo_demand"] / per_sku["base_demand"] - 1) * 100, 0.0).round(1)
    per_sku["promo_roi_pct"] = np.where(
        per_sku["promo_cost"] > 0,
        (per_sku["inc_profit"] - per_sku["promo_cost"]) / per_sku["promo_cost"] * 100,
        0.0).round(1)

    total_inc = float(promo["inc_profit"].sum())
    total_cost = float(promo["promo_cost"].sum())
    portfolio = {
        "promo_roi_pct": round((total_inc - total_cost) / total_cost * 100, 1) if total_cost else 0.0,
        "promo_incremental_profit": round(total_inc, 0),
        "promo_discount_cost": round(total_cost, 0),
        "avg_promo_lift_pct": round(float(per_sku["promo_lift_pct"].mean()), 1),
    }
    return per_sku[keys + ["promo_lift_pct", "promo_roi_pct"]], portfolio


def compute_financials(inv, stock, raw, gross_margin_rate=0.35, holding_rate=0.25):
    """Build the per-SKU financial table and the portfolio summary."""
    keys = ["Store ID", "Product ID"]
    df = inv.merge(
        stock[keys + ["avg_inventory_naive", "avg_inventory_model"]], on=keys, how="left")

    df["annual_demand"] = (df["avg_daily_demand"] * DAYS_PER_YEAR).round(0)
    df["revenue"] = df["annual_revenue"].round(0)
    df["gross_margin_pct"] = round(gross_margin_rate * 100, 1)
    df["gross_profit"] = (df["revenue"] * gross_margin_rate).round(0)

    # Inventory-reduction savings: fewer units held -> freed capital -> holding saved.
    df["inventory_reduction_units"] = (df["avg_inventory_naive"] - df["avg_inventory_model"]).round(1)
    df["capital_freed"] = (df["inventory_reduction_units"] * df["unit_price"]).round(0)
    df["holding_savings"] = (df["capital_freed"] * holding_rate).round(0)

    # Turnover = annual demand / average inventory carried under the model.
    df["turnover"] = np.where(
        df["avg_inventory_model"] > 0,
        df["annual_demand"] / df["avg_inventory_model"], np.nan).round(1)

    promo_sku, promo_summary = promotion_roi(raw, gross_margin_rate)
    df = df.merge(promo_sku, on=keys, how="left")
    df["promo_lift_pct"] = df["promo_lift_pct"].fillna(0.0)
    df["promo_roi_pct"] = df["promo_roi_pct"].fillna(0.0)

    cols = keys + ["abc_class", "annual_demand", "unit_price", "revenue",
                   "gross_margin_pct", "gross_profit", "avg_inventory_model",
                   "inventory_reduction_units", "capital_freed", "holding_savings",
                   "turnover", "promo_lift_pct", "promo_roi_pct"]
    table = df[cols].sort_values("revenue", ascending=False).reset_index(drop=True)

    naive = float(df["avg_inventory_naive"].sum())
    model = float(df["avg_inventory_model"].sum())
    total_rev = float(df["revenue"].sum())
    summary = {
        "total_revenue": round(total_rev, 0),
        "total_gross_profit": round(float(df["gross_profit"].sum()), 0),
        "gross_margin_pct": round(gross_margin_rate * 100, 1),
        "total_capital_freed": round(float(df["capital_freed"].sum()), 0),
        "annual_holding_savings": round(float(df["holding_savings"].sum()), 0),
        "inventory_reduction_pct": round((naive - model) / naive * 100, 1) if naive else 0.0,
        "portfolio_turnover": round(df["annual_demand"].sum() / model, 1) if model else 0.0,
        "n_skus": int(len(df)),
        "assumptions": {"gross_margin_rate": gross_margin_rate, "holding_rate": holding_rate},
    }
    summary.update(promo_summary)
    return table, summary


def _plot(table, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_abc = table.groupby("abc_class")[["revenue", "gross_profit"]].sum().reindex(["A", "B", "C"])
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(by_abc))
    ax.bar(x - 0.2, by_abc["revenue"] / 1e6, 0.4, label="Revenue (₺M)", color="#94a3b8")
    ax.bar(x + 0.2, by_abc["gross_profit"] / 1e6, 0.4, label="Gross profit (₺M)", color="#2563eb")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Class {c}" for c in by_abc.index])
    ax.set_ylabel("₺ million")
    ax.set_title("Revenue and gross profit by ABC class")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main(gross_margin_rate=0.35, holding_rate=0.25):
    inv = pd.read_csv(INV_PATH)
    stock = pd.read_csv(STOCK_PATH)
    raw = pd.read_csv(DATA_PATH)
    table, summary = compute_financials(inv, stock, raw, gross_margin_rate, holding_rate)

    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULTS_DIR / "financial_metrics.csv", index=False)
    with open(RESULTS_DIR / "financial_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    _plot(table, RESULTS_DIR / "19_financial_summary.png")

    print(f"SKUs: {summary['n_skus']}  |  gross margin assumed: {summary['gross_margin_pct']}%")
    print(f"Revenue        : ₺{summary['total_revenue']:,.0f}")
    print(f"Gross profit   : ₺{summary['total_gross_profit']:,.0f}")
    print(f"Capital freed  : ₺{summary['total_capital_freed']:,.0f} "
          f"(inventory down {summary['inventory_reduction_pct']}%)")
    print(f"Holding saved  : ₺{summary['annual_holding_savings']:,.0f}/yr")
    print(f"Turnover       : {summary['portfolio_turnover']}x")
    print(f"Promotion ROI  : {summary['promo_roi_pct']}%  "
          f"(avg lift {summary['avg_promo_lift_pct']}%)")
    print(f"\nSaved -> {RESULTS_DIR/'financial_metrics.csv'}, "
          f"{RESULTS_DIR/'financial_summary.json'}, {RESULTS_DIR/'19_financial_summary.png'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Financial-metric layer for the demand/inventory analytics")
    p.add_argument("--gross-margin", type=float, default=0.35,
                   help="gross margin as a fraction of revenue")
    p.add_argument("--holding-rate", type=float, default=0.25,
                   help="annual holding cost as a fraction of unit price")
    args = p.parse_args()
    main(gross_margin_rate=args.gross_margin, holding_rate=args.holding_rate)
