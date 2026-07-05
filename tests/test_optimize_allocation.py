"""Tests for the multi-store stock-allocation LP (optimize_allocation.py)."""

import numpy as np
import pandas as pd
import pytest

from optimize_allocation import (
    ABC_WEIGHT, optimize_allocation, sensitivity_analysis, service_target)


def _demo_frame():
    """Small synthetic SKU set: two A, two C, mixed prices and demand."""
    return pd.DataFrame({
        "Store ID": ["S1", "S1", "S2", "S2"],
        "Product ID": ["P1", "P2", "P3", "P4"],
        "abc_class": ["A", "A", "C", "C"],
        "mu": [100.0, 120.0, 80.0, 60.0],
        "sigma": [10.0, 12.0, 8.0, 6.0],
        "price": [50.0, 40.0, 20.0, 30.0],
        "current": [10, 20, 5, 5],
    }).assign(
        tau=lambda d: service_target(d["mu"], d["sigma"], 1.64),
        w=lambda d: d["abc_class"].map(ABC_WEIGHT),
    )


def test_service_target_adds_safety_buffer():
    assert service_target(100, 10, 1.64) == pytest.approx(116.4)
    assert service_target(100, 0, 1.64) == 100  # no variability -> just the mean


def test_solver_reaches_optimal_and_respects_resource_limits():
    df = _demo_frame()
    full_cost = float((df["price"] * df["tau"]).sum())
    budget = 0.75 * full_cost
    capacity = int(0.9 * df["tau"].sum())
    out, summary = optimize_allocation(df, budget=budget, capacity=capacity,
                                       min_service=0.5)
    assert summary["solver_status"] == "Optimal"
    # Budget and capacity constraints must hold (tiny numeric tolerance).
    assert summary["budget_used"] <= budget + 1e-6
    assert summary["capacity_used"] <= capacity


def test_allocation_within_min_service_and_target_bounds():
    df = _demo_frame()
    full_cost = float((df["price"] * df["tau"]).sum())
    out, _ = optimize_allocation(df, budget=0.7 * full_cost,
                                 capacity=10**9, min_service=0.5)
    merged = out.merge(df[["Store ID", "Product ID", "tau"]],
                       on=["Store ID", "Product ID"])
    # Every allocation sits between the min-service floor and the full target.
    assert (merged["allocation"] <= np.ceil(merged["tau"]) + 1).all()
    assert (merged["allocation"] >= np.floor(0.5 * merged["tau"]) - 1).all()


def test_generous_budget_serves_everyone_fully():
    df = _demo_frame()
    out, summary = optimize_allocation(df, budget=10**9, capacity=10**9,
                                       min_service=0.5)
    assert summary["skus_at_full_target"] == len(df)
    assert summary["stockout_penalty"] == pytest.approx(0.0, abs=1.0)
    assert summary["avg_service_fill_pct"] == pytest.approx(100.0, abs=0.5)


def test_tighter_budget_allocates_fewer_units():
    df = _demo_frame()
    full_cost = float((df["price"] * df["tau"]).sum())
    _, loose = optimize_allocation(df, budget=0.9 * full_cost, capacity=10**9)
    _, tight = optimize_allocation(df, budget=0.5 * full_cost, capacity=10**9)
    assert tight["capacity_used"] < loose["capacity_used"]


def test_scarce_budget_protects_A_class_over_C_class():
    """Under a tight budget the penalty weighting should keep A-class SKUs
    better served than cheap C-class ones."""
    df = _demo_frame()
    full_cost = float((df["price"] * df["tau"]).sum())
    out, _ = optimize_allocation(df, budget=0.6 * full_cost, capacity=10**9,
                                 min_service=0.3)
    a_fill = out.loc[out["abc_class"] == "A", "service_fill_pct"].mean()
    c_fill = out.loc[out["abc_class"] == "C", "service_fill_pct"].mean()
    assert a_fill > c_fill


def test_optimizer_beats_even_cut_baseline():
    df = _demo_frame()
    full_cost = float((df["price"] * df["tau"]).sum())
    _, summary = optimize_allocation(df, budget=0.6 * full_cost, capacity=10**9,
                                     min_service=0.3)
    assert summary["savings_vs_baseline"] >= -1e-6      # never worse than baseline


# --- Feasibility / sensitivity -------------------------------------------------

def test_infeasible_below_the_budget_floor():
    df = _demo_frame()
    full_cost = float((df["price"] * df["tau"]).sum())
    # Floor to afford the min-service lower bounds = 0.5 * full_cost.
    _, tight = optimize_allocation(df, budget=0.30 * full_cost, capacity=10**9,
                                   min_service=0.5)
    assert tight["solver_status"] == "Infeasible"
    _, ok = optimize_allocation(df, budget=0.55 * full_cost, capacity=10**9,
                                min_service=0.5)
    assert ok["solver_status"] == "Optimal"


def test_min_feasible_thresholds_match_min_service_floors():
    df = _demo_frame()
    full_cost = float((df["price"] * df["tau"]).sum())
    full_units = float(df["tau"].sum())
    _, s = optimize_allocation(df, budget=0.8 * full_cost, capacity=0.9 * full_units,
                               min_service=0.5)
    assert s["min_feasible_budget"] == pytest.approx(0.5 * full_cost, rel=1e-6)
    assert s["min_feasible_capacity"] == pytest.approx(0.5 * full_units, abs=1)


def test_binding_budget_has_positive_shadow_price():
    df = _demo_frame()
    full_cost = float((df["price"] * df["tau"]).sum())
    _, tight = optimize_allocation(df, budget=0.7 * full_cost, capacity=10**9)
    assert tight["budget_binding"] is True
    assert tight["budget_shadow_price"] > 0
    # A generous budget leaves the constraint slack -> zero shadow price.
    _, loose = optimize_allocation(df, budget=10 * full_cost, capacity=10**9)
    assert loose["budget_shadow_price"] == pytest.approx(0.0, abs=1e-6)
    assert loose["budget_binding"] is False


def test_sensitivity_curve_cost_falls_as_budget_grows():
    df = _demo_frame()
    full_cost = float((df["price"] * df["tau"]).sum())
    sens = sensitivity_analysis(df, budget=0.8 * full_cost, capacity=10**9,
                                min_service=0.5, n_points=7)
    feas = [c for c in sens["curve"] if c["total_cost"] is not None]
    costs = [c["total_cost"] for c in feas]
    assert costs == sorted(costs, reverse=True)          # more budget -> lower cost
    assert any(c["status"] == "Infeasible" for c in sens["curve"])  # floor is crossed
    assert sens["min_feasible_budget"] == pytest.approx(0.5 * full_cost, rel=1e-6)
