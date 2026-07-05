r"""
Multi-store stock-allocation optimisation with a real mathematical program.

The demand model and the classic EOQ / safety-stock analytics tell you what each
SKU *ideally* wants. In practice you cannot give every SKU its full target: the
procurement budget and the warehouse are finite. This module turns that trade-off
into a linear program and solves it with PuLP + CBC.

--------------------------------------------------------------------------------
DECISION
    x_i  >= 0    units of stock to allocate to store-product i  (one var per SKU)

DERIVED PER SKU
    mu_i   = avg_daily_demand_i * H            expected demand over the horizon H
    sig_i  = demand_std_i * sqrt(H)            demand std over the horizon
    tau_i  = mu_i + z * sig_i                  service target = cover mean + safety
    h_i    = holding_rate * price_i * H/365     holding cost per unit over horizon
    w_i    = ABC weight (A>B>C)                 how much a lost sale here hurts
    b_i    = penalty_mult * price_i * w_i        stockout penalty per unit short

OBJECTIVE   (minimise total expected cost)
    min  sum_i ( h_i * x_i  +  b_i * short_i )
         \_______/    \________/
          holding      stockout penalty

CONSTRAINTS
    short_i >= tau_i - x_i,   short_i >= 0        shortfall below the service target
    min_service * tau_i <= x_i <= tau_i           minimum service level per SKU
    sum_i price_i * x_i <= budget                 total procurement budget
    sum_i x_i           <= capacity               warehouse capacity (units)

Because b_i >> h_i, the unconstrained optimum sends every SKU to its target tau_i.
The budget / capacity limits are what force the model to *allocate*: it protects
high-value A-class SKUs at full service and sacrifices cheap C-class coverage
first, which is exactly what a planner would do by hand — but proven optimal.

Outputs:
    results/optimization_allocation.csv   per-SKU allocation + costs
    results/optimization_summary.json     objective, budget/capacity use, service
    results/18_optimization_allocation.png bar chart of the allocation

Run:
    .venv/bin/python optimize_allocation.py
    .venv/bin/python optimize_allocation.py --budget-frac 0.7 --min-service 0.6
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

RESULTS_DIR = Path("results")
INV_PATH = RESULTS_DIR / "inventory_analytics.csv"
STOCK_PATH = RESULTS_DIR / "stock_recommendations.csv"

# ABC weight: a lost sale on an A-class SKU costs the business far more than on a
# C-class one, so the penalty term is scaled by class importance.
ABC_WEIGHT = {"A": 1.0, "B": 0.6, "C": 0.35}


def service_target(mu, sigma, z):
    """Order-up-to target: cover expected demand plus a z-scaled safety buffer."""
    return mu + z * sigma


def load_skus(inv_path=INV_PATH, stock_path=STOCK_PATH, horizon=7, service_z=1.64):
    """Merge the policy table with demand variability into an allocation frame.

    Returns a DataFrame with one row per store-product and the derived
    per-SKU parameters (mu, sigma, tau, price, current, abc weight).
    """
    inv = pd.read_csv(inv_path)
    stock = pd.read_csv(stock_path)
    keys = ["Store ID", "Product ID"]
    df = inv.merge(stock[keys + ["demand_std"]], on=keys, how="left")
    df["demand_std"] = df["demand_std"].fillna(0.0)

    df["mu"] = df["avg_daily_demand"] * horizon
    df["sigma"] = df["demand_std"] * np.sqrt(horizon)
    df["tau"] = service_target(df["mu"], df["sigma"], service_z)
    df["price"] = df["unit_price"].astype(float)
    df["current"] = df["current_inventory"].astype(float)
    df["w"] = df["abc_class"].map(ABC_WEIGHT).fillna(ABC_WEIGHT["C"])
    return df


def optimize_allocation(df, budget, capacity, min_service=0.5,
                        holding_rate=0.25, penalty_mult=1.5, horizon=7):
    """Build and solve the allocation LP. Returns (result_df, summary).

    Parameters
    ----------
    df          frame from load_skus (needs mu, sigma, tau, price, current, w).
    budget      max total procurement spend  sum price_i * x_i.
    capacity    max total units in the warehouse  sum x_i.
    min_service fraction of each SKU's target tau that must be met (0..1).
    holding_rate annual holding cost as a fraction of unit price.
    penalty_mult stockout penalty as a multiple of unit price (x ABC weight).
    """
    n = len(df)
    price = df["price"].to_numpy()
    tau = df["tau"].to_numpy()
    w = df["w"].to_numpy()
    h = holding_rate * price * (horizon / 365.0)          # holding cost / unit
    b = penalty_mult * price * w                          # stockout penalty / unit

    prob = pulp.LpProblem("multi_store_stock_allocation", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{i}", lowBound=min_service * tau[i], upBound=tau[i])
         for i in range(n)]
    short = [pulp.LpVariable(f"short_{i}", lowBound=0) for i in range(n)]

    # Objective: holding cost on what we carry + penalty on the service shortfall.
    prob += pulp.lpSum(h[i] * x[i] + b[i] * short[i] for i in range(n))

    # short_i captures how far below the target we fell.
    for i in range(n):
        prob += short[i] >= tau[i] - x[i]

    # Shared resource limits — the reason allocation is non-trivial.
    prob += pulp.lpSum(price[i] * x[i] for i in range(n)) <= budget, "budget"
    prob += pulp.lpSum(x[i] for i in range(n)) <= capacity, "capacity"

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[prob.status]

    # Shadow prices (constraint duals): ₺ of total cost saved per one extra unit
    # of the resource. A binding constraint has a positive shadow price; a slack
    # one is 0 (relaxing it changes nothing).
    def _shadow(name):
        try:
            pi = prob.constraints[name].pi
            return (round(-float(pi), 4) or 0.0) if pi is not None else 0.0  # avoid -0.0
        except Exception:
            return 0.0
    budget_shadow = _shadow("budget") if status == "Optimal" else 0.0
    capacity_shadow = _shadow("capacity") if status == "Optimal" else 0.0

    alloc = np.array([x[i].value() if x[i].value() is not None else 0.0
                      for i in range(n)])
    alloc_units = np.round(alloc).astype(int)
    shortfall = np.maximum(tau - alloc, 0.0)

    out = df[["Store ID", "Product ID", "abc_class"]].copy()
    out["price"] = np.round(price, 2)
    out["expected_demand"] = np.round(df["mu"].to_numpy(), 1)
    out["service_target"] = np.round(tau, 1)
    out["allocation"] = alloc_units
    out["current_inventory"] = df["current"].astype(int).to_numpy()
    out["recommended_order"] = np.maximum(alloc_units - out["current_inventory"], 0)
    out["service_fill_pct"] = np.round(np.where(tau > 0, alloc / tau, 1.0) * 100, 1)
    out["holding_cost"] = np.round(h * alloc, 2)
    out["stockout_penalty"] = np.round(b * shortfall, 2)
    out = out.sort_values(["abc_class", "recommended_order"],
                          ascending=[True, False]).reset_index(drop=True)

    # Baseline: shrink every target proportionally to fit the budget (naive,
    # class-blind). Shows what the optimiser buys you over an even cut.
    full_cost = float((price * tau).sum())
    s = min(1.0, budget / full_cost) if full_cost > 0 else 1.0
    base_alloc = np.maximum(np.minimum(tau, s * tau), min_service * tau)
    base_short = np.maximum(tau - base_alloc, 0.0)
    base_cost = float((h * base_alloc + b * base_short).sum())
    opt_cost = float((h * alloc + b * shortfall).sum())

    summary = {
        "solver_status": status,
        "n_skus": int(n),
        "budget": round(float(budget), 2),
        "budget_used": round(float((price * alloc).sum()), 2),
        "capacity": int(capacity),
        "capacity_used": int(alloc_units.sum()),
        "total_cost": round(opt_cost, 2),
        "holding_cost": round(float((h * alloc).sum()), 2),
        "stockout_penalty": round(float((b * shortfall).sum()), 2),
        "avg_service_fill_pct": round(float(np.where(tau > 0, alloc / tau, 1.0).mean() * 100), 1),
        "skus_at_full_target": int((alloc >= tau - 0.5).sum()),
        "min_service_pct": round(min_service * 100, 1),
        "baseline_cost": round(base_cost, 2),
        "savings_vs_baseline": round(base_cost - opt_cost, 2),
        "savings_pct": round((base_cost - opt_cost) / base_cost * 100, 1) if base_cost > 0 else 0.0,
        # Feasibility & sensitivity
        "budget_shadow_price": budget_shadow,
        "capacity_shadow_price": capacity_shadow,
        "budget_binding": bool(budget_shadow > 1e-6),
        "capacity_binding": bool(capacity_shadow > 1e-6),
        "min_feasible_budget": round(min_service * full_cost, 2),
        "min_feasible_capacity": int(np.ceil(min_service * float(tau.sum()))),
    }
    return out, summary


def sensitivity_analysis(df, budget, capacity, min_service=0.5, holding_rate=0.25,
                         penalty_mult=1.5, horizon=7, n_points=7):
    """Trace how the optimum responds as the budget is scaled, and pin down the
    infeasibility thresholds and shadow prices.

    Returns a dict with the two feasibility floors (below which no plan exists),
    the base-case shadow prices, and a budget-sweep curve of total cost, service
    and the budget shadow price at each level.
    """
    price = df["price"].to_numpy()
    tau = df["tau"].to_numpy()
    full_cost = float((price * tau).sum())
    full_units = float(tau.sum())
    min_budget = min_service * full_cost
    min_units = min_service * full_units

    # Sweep from just below the feasibility floor up to the full-target cost.
    fracs = np.linspace(0.4, 1.0, n_points)
    curve = []
    for f in fracs:
        b = f * full_cost
        _, s = optimize_allocation(df, budget=b, capacity=capacity,
                                   min_service=min_service, holding_rate=holding_rate,
                                   penalty_mult=penalty_mult, horizon=horizon)
        feasible = s["solver_status"] == "Optimal"
        curve.append({
            "budget_frac": round(float(f), 3),
            "budget": round(b, 0),
            "status": s["solver_status"],
            "total_cost": s["total_cost"] if feasible else None,
            "avg_service_fill_pct": s["avg_service_fill_pct"] if feasible else None,
            "budget_shadow_price": s["budget_shadow_price"] if feasible else None,
        })

    _, base = optimize_allocation(df, budget=budget, capacity=capacity,
                                  min_service=min_service, holding_rate=holding_rate,
                                  penalty_mult=penalty_mult, horizon=horizon)
    return {
        "min_feasible_budget": round(min_budget, 2),
        "min_feasible_capacity": int(np.ceil(min_units)),
        "budget_shadow_price": base["budget_shadow_price"],
        "capacity_shadow_price": base["capacity_shadow_price"],
        "budget_binding": base["budget_binding"],
        "capacity_binding": base["capacity_binding"],
        "curve": curve,
    }


def _plot(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = out.sort_values("recommended_order", ascending=False).head(15)[::-1]
    labels = top["Store ID"] + "·" + top["Product ID"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels, top["allocation"], color="#2563eb", label="Optimised allocation")
    ax.barh(labels, top["current_inventory"], color="#94a3b8", label="Current stock")
    ax.set_xlabel("Units")
    ax.set_title("Optimised stock allocation vs current — top reorder needs")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _plot_sensitivity(sens, summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feas = [c for c in sens["curve"] if c["total_cost"] is not None]
    infeas = [c for c in sens["curve"] if c["total_cost"] is None]
    bx = [c["budget"] / 1e6 for c in feas]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(bx, [c["total_cost"] / 1e6 for c in feas], "o-", color="#2563eb",
            label="Total expected cost (₺M)")
    ax.set_xlabel("Budget (₺M)")
    ax.set_ylabel("Total expected cost (₺M)", color="#2563eb")
    ax.axvline(summary["min_feasible_budget"] / 1e6, ls="--", color="#e05a44",
               label="Feasibility floor")
    for c in infeas:
        ax.axvspan(c["budget"] / 1e6 - 0.05, c["budget"] / 1e6 + 0.05, color="#e05a44", alpha=0.08)
    ax2 = ax.twinx()
    ax2.plot(bx, [c["avg_service_fill_pct"] for c in feas], "s--", color="#94a3b8",
             label="Service fill %")
    ax2.set_ylabel("Service fill %", color="#94a3b8")
    ax.set_title("Sensitivity: cost & service vs budget (infeasible below the floor)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main(budget_frac=0.8, capacity_frac=0.9, min_service=0.5,
         holding_rate=0.25, penalty_mult=1.5, horizon=7, service_z=1.64):
    df = load_skus(horizon=horizon, service_z=service_z)
    full_cost = float((df["price"] * df["tau"]).sum())
    full_units = float(df["tau"].sum())
    budget = budget_frac * full_cost
    capacity = capacity_frac * full_units

    out, summary = optimize_allocation(
        df, budget=budget, capacity=capacity, min_service=min_service,
        holding_rate=holding_rate, penalty_mult=penalty_mult, horizon=horizon)
    summary["horizon_days"] = horizon
    summary["full_target_cost"] = round(full_cost, 2)

    # Feasibility / sensitivity: thresholds, shadow prices, and a budget sweep.
    sens = sensitivity_analysis(df, budget=budget, capacity=capacity,
                                min_service=min_service, holding_rate=holding_rate,
                                penalty_mult=penalty_mult, horizon=horizon)
    summary["sensitivity"] = sens["curve"]

    RESULTS_DIR.mkdir(exist_ok=True)
    out.to_csv(RESULTS_DIR / "optimization_allocation.csv", index=False)
    pd.DataFrame(sens["curve"]).to_csv(RESULTS_DIR / "optimization_sensitivity.csv", index=False)
    with open(RESULTS_DIR / "optimization_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    _plot(out, RESULTS_DIR / "18_optimization_allocation.png")
    _plot_sensitivity(sens, summary, RESULTS_DIR / "20_optimization_sensitivity.png")

    print(f"Solver: {summary['solver_status']}  |  SKUs: {summary['n_skus']}")
    print(f"Budget {summary['budget']:,.0f} -> used {summary['budget_used']:,.0f} "
          f"({summary['budget_used']/summary['budget']*100:.0f}%)")
    print(f"Capacity {summary['capacity']:,} units -> used {summary['capacity_used']:,}")
    print(f"Total expected cost: {summary['total_cost']:,.0f} "
          f"(holding {summary['holding_cost']:,.0f} + penalty {summary['stockout_penalty']:,.0f})")
    print(f"Avg service fill: {summary['avg_service_fill_pct']}%  |  "
          f"{summary['skus_at_full_target']}/{summary['n_skus']} SKUs at full target")
    print(f"Savings vs even-cut baseline: {summary['savings_vs_baseline']:,.0f} "
          f"({summary['savings_pct']}%)")
    print(f"Budget shadow price: ₺{summary['budget_shadow_price']:.3f} saved per +₺1 budget "
          f"({'binding' if summary['budget_binding'] else 'slack'})  |  "
          f"capacity shadow: {summary['capacity_shadow_price']:.3f} "
          f"({'binding' if summary['capacity_binding'] else 'slack'})")
    print(f"Feasibility floor: budget >= ₺{summary['min_feasible_budget']:,.0f}, "
          f"capacity >= {summary['min_feasible_capacity']:,} units (else infeasible)")
    print(f"\nSaved -> {RESULTS_DIR/'optimization_allocation.csv'}, "
          f"{RESULTS_DIR/'optimization_summary.json'}, "
          f"{RESULTS_DIR/'optimization_sensitivity.csv'}, "
          f"{RESULTS_DIR/'18_optimization_allocation.png'}, "
          f"{RESULTS_DIR/'20_optimization_sensitivity.png'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Multi-store stock-allocation LP (PuLP/CBC)")
    p.add_argument("--budget-frac", type=float, default=0.8,
                   help="budget as a fraction of the full service-target cost")
    p.add_argument("--capacity-frac", type=float, default=0.9,
                   help="capacity as a fraction of total target units")
    p.add_argument("--min-service", type=float, default=0.5,
                   help="minimum service level: fraction of each SKU's target that must be met")
    p.add_argument("--holding-rate", type=float, default=0.25,
                   help="annual holding cost as a fraction of unit price")
    p.add_argument("--penalty-mult", type=float, default=1.5,
                   help="stockout penalty as a multiple of unit price")
    p.add_argument("--horizon", type=int, default=7, help="planning horizon in days")
    args = p.parse_args()
    main(budget_frac=args.budget_frac, capacity_frac=args.capacity_frac,
         min_service=args.min_service, holding_rate=args.holding_rate,
         penalty_mult=args.penalty_mult, horizon=args.horizon)
