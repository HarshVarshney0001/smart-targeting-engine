import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- PATHS ----------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

SEGMENTATION_CSV_PATH = MODEL_DIR / "segmentation_results.csv"
OPTIMIZED_LIST_PATH = MODEL_DIR / "optimized_targeting_list.csv"
BUDGET_CURVE_PLOT_PATH = MODEL_DIR / "budget_optimization_curve.png"
SCENARIO_TABLE_PATH = MODEL_DIR / "budget_scenarios.csv"

# ---------- ASSUMPTIONS (documented, illustrative — NOT real cost data) ----------
# Hillstrom's "channel" column is the customer's historical purchase channel
# (Phone / Web / Multichannel) — not the treatment channel (all treatment in
# this dataset is email-based). We use it here as a SYNTHETIC cost-driver to
# demonstrate variable-cost budget optimization: customers whose past channel
# preference is Phone are assumed costlier to re-engage than Web or
# Multichannel customers (baseline category dropped by get_dummies).
COST_MULTICHANNEL_BASELINE = 0.10   # baseline category (channel_Phone=0, channel_Web=0)
COST_WEB = 0.15                     # customer historically used Web
COST_PHONE = 0.50                   # customer historically used Phone (assumed costliest to re-engage)

BUDGET_FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]  # % of eligible customers simulated


# ---------- 1. LOAD SEGMENTATION RESULTS (output of Step 5) ----------
def load_segmentation_results():
    df = pd.read_csv(SEGMENTATION_CSV_PATH)
    print("Loaded segmentation results:", df.shape)
    print("Columns:", list(df.columns))
    return df


# ---------- 2. ASSIGN VARIABLE COST PER CUSTOMER ----------
def assign_cost(df):
    df = df.copy()
    df["cost"] = np.where(
        df["channel_Phone"] == 1, COST_PHONE,
        np.where(df["channel_Web"] == 1, COST_WEB, COST_MULTICHANNEL_BASELINE)
    )

    print("\nCost distribution by channel:")
    print(df["cost"].value_counts().sort_index())
    return df


# ---------- 3. RANK CUSTOMERS BY UPLIFT-PER-COST ----------
def rank_by_roi(df):
    df = df.copy()
    df["roi_score"] = df["uplift_score"] / df["cost"]

    df = df.sort_values("roi_score", ascending=False).reset_index(drop=True)

    df["cumulative_customers"] = np.arange(1, len(df) + 1)
    df["cumulative_cost"] = df["cost"].cumsum()
    df["cumulative_incremental_conversions"] = df["uplift_score"].cumsum()

    print("\nCustomers ranked by ROI (uplift per $ cost, variable cost by channel).")
    return df


# ---------- 4. FIND NATURAL BREAK-EVEN POINT (no budget constraint needed) ----------
def find_breakeven_point(ranked_df):
    positive_uplift_df = ranked_df[ranked_df["uplift_score"] > 0]
    breakeven_count = len(positive_uplift_df)
    breakeven_cost = positive_uplift_df["cost"].sum()
    breakeven_gain = positive_uplift_df["uplift_score"].sum()

    print("\n" + "=" * 60)
    print("NATURAL BREAK-EVEN POINT (targeting stops making sense beyond this)")
    print("=" * 60)
    print(f"Customers with positive uplift: {breakeven_count} ({breakeven_count / len(ranked_df) * 100:.2f}% of base)")
    print(f"Cost to reach this point: ${breakeven_cost:,.2f}")
    print(f"Expected incremental conversions at this point: {breakeven_gain:.2f}")

    return breakeven_count, breakeven_cost, breakeven_gain


# ---------- 5. SIMULATE MULTIPLE BUDGET SCENARIOS (based on $ cost now, not just count) ----------
def simulate_budget_scenarios(ranked_df):
    total_cost_if_all_targeted = ranked_df["cost"].sum()
    rows = []

    for frac in BUDGET_FRACTIONS:
        budget_cap = total_cost_if_all_targeted * frac
        subset = ranked_df[ranked_df["cumulative_cost"] <= budget_cap]

        rows.append({
            "budget_fraction": f"{int(frac * 100)}%",
            "budget_cap": round(budget_cap, 2),
            "customers_targeted": len(subset),
            "actual_cost_spent": round(subset["cost"].sum(), 2),
            "expected_incremental_conversions": round(subset["uplift_score"].sum(), 2),
            "avg_uplift_of_targeted": round(subset["uplift_score"].mean(), 4),
        })

    scenario_df = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print("BUDGET SCENARIOS (variable cost by channel)")
    print("=" * 60)
    print(scenario_df.to_string(index=False))

    scenario_df.to_csv(SCENARIO_TABLE_PATH, index=False)
    print(f"\nBudget scenario table saved to: {SCENARIO_TABLE_PATH}")
    return scenario_df


# ---------- 6. PLOT: CUMULATIVE INCREMENTAL CONVERSIONS VS CUMULATIVE $ SPENT ----------
def plot_budget_curve(ranked_df, breakeven_count, breakeven_cost):
    plt.figure(figsize=(9, 6))
    plt.plot(ranked_df["cumulative_cost"], ranked_df["cumulative_incremental_conversions"],
              linewidth=2, color="#1f77b4")

    breakeven_gain = ranked_df.iloc[breakeven_count - 1]["cumulative_incremental_conversions"]
    plt.axvline(breakeven_cost, linestyle="--", color="#d62728",
                label=f"Break-even point (${breakeven_cost:,.0f} spent)")
    plt.scatter([breakeven_cost], [breakeven_gain], color="#d62728", zorder=5)

    plt.xlabel("Cumulative $ spent (ranked by uplift-per-cost, variable cost by channel)")
    plt.ylabel("Cumulative expected incremental conversions")
    plt.title("Budget Optimization Curve — Smart Targeting Engine (Variable Cost)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(BUDGET_CURVE_PLOT_PATH, dpi=150)
    print(f"\nBudget optimization curve saved to: {BUDGET_CURVE_PLOT_PATH}")
    plt.close()


# ---------- 7. SAVE FINAL RECOMMENDED TARGETING LIST (break-even based) ----------
def save_optimized_list(ranked_df, breakeven_count):
    optimized_list = ranked_df.iloc[:breakeven_count].copy()
    optimized_list.to_csv(OPTIMIZED_LIST_PATH, index=False)
    print(f"\nOptimized targeting list ({breakeven_count} customers) saved to: {OPTIMIZED_LIST_PATH}")
    return optimized_list


# ---------- MAIN PIPELINE ----------
if __name__ == "__main__":
    df = load_segmentation_results()
    df = assign_cost(df)
    ranked_df = rank_by_roi(df)

    breakeven_count, breakeven_cost, breakeven_gain = find_breakeven_point(ranked_df)

    scenario_df = simulate_budget_scenarios(ranked_df)

    plot_budget_curve(ranked_df, breakeven_count, breakeven_cost)

    optimized_list = save_optimized_list(ranked_df, breakeven_count)