"""Tests for the financial-metric layer (financial_metrics.py)."""

import numpy as np
import pandas as pd
import pytest

from financial_metrics import compute_financials, promotion_roi


def _inv():
    return pd.DataFrame({
        "Store ID": ["S1", "S2"],
        "Product ID": ["P1", "P2"],
        "abc_class": ["A", "B"],
        "annual_revenue": [1_000_000.0, 400_000.0],
        "unit_price": [50.0, 20.0],
        "avg_daily_demand": [100.0, 60.0],
    })


def _stock():
    return pd.DataFrame({
        "Store ID": ["S1", "S2"],
        "Product ID": ["P1", "P2"],
        "avg_inventory_naive": [500.0, 300.0],
        "avg_inventory_model": [350.0, 240.0],
    })


def _raw(promo_demand=200, base_demand=100, discount=10):
    """Two SKUs, each with promo and non-promo days."""
    rows = []
    for sku, (s, p, price) in {"P1": ("S1", "P1", 50.0), "P2": ("S2", "P2", 20.0)}.items():
        for _ in range(10):
            rows.append({"Store ID": s, "Product ID": p, "Promotion": 0,
                         "Discount": 0, "Price": price, "Demand": base_demand})
            rows.append({"Store ID": s, "Product ID": p, "Promotion": 1,
                         "Discount": discount, "Price": price, "Demand": promo_demand})
    return pd.DataFrame(rows)


def test_gross_profit_uses_margin_rate():
    table, summary = compute_financials(_inv(), _stock(), _raw(), gross_margin_rate=0.35)
    row = table[table["Product ID"] == "P1"].iloc[0]
    assert row["gross_profit"] == pytest.approx(1_000_000 * 0.35)
    assert summary["total_gross_profit"] == pytest.approx(1_400_000 * 0.35)


def test_holding_savings_from_inventory_reduction():
    table, summary = compute_financials(_inv(), _stock(), _raw(),
                                        gross_margin_rate=0.35, holding_rate=0.25)
    p1 = table[table["Product ID"] == "P1"].iloc[0]
    # (500 - 350) units * ₺50 = ₺7,500 freed; * 0.25 = ₺1,875 saved.
    assert p1["capital_freed"] == pytest.approx(7500)
    assert p1["holding_savings"] == pytest.approx(1875)
    assert summary["inventory_reduction_pct"] == pytest.approx((800 - 590) / 800 * 100, abs=0.1)


def test_turnover_is_annual_demand_over_avg_inventory():
    table, _ = compute_financials(_inv(), _stock(), _raw())
    p1 = table[table["Product ID"] == "P1"].iloc[0]
    assert p1["turnover"] == pytest.approx(100 * 365 / 350, abs=0.1)


def test_promotion_roi_positive_when_lift_is_cheap():
    # Big lift (100 -> 200), tiny 2% discount -> promo clearly pays off.
    _, portfolio = promotion_roi(_raw(promo_demand=200, base_demand=100, discount=2),
                                 gross_margin_rate=0.35)
    assert portfolio["avg_promo_lift_pct"] == pytest.approx(100.0, abs=0.1)
    assert portfolio["promo_roi_pct"] > 0


def test_promotion_roi_negative_without_lift():
    # No lift but a discount is still handed out -> ROI must be negative.
    _, portfolio = promotion_roi(_raw(promo_demand=100, base_demand=100, discount=10),
                                 gross_margin_rate=0.35)
    assert portfolio["avg_promo_lift_pct"] == pytest.approx(0.0, abs=0.1)
    assert portfolio["promo_roi_pct"] < 0


def test_summary_totals_match_row_sums():
    table, summary = compute_financials(_inv(), _stock(), _raw())
    assert summary["total_revenue"] == pytest.approx(table["revenue"].sum())
    assert summary["total_capital_freed"] == pytest.approx(table["capital_freed"].sum())
    assert summary["n_skus"] == 2
