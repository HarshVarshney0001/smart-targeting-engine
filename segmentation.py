import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

# ---------- PATHS ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "hillstrom.csv"
MODEL_DIR = BASE_DIR / "models"

T_LEARNER_PATH = MODEL_DIR / "t_learner.pkl"
BASELINE_MODEL_PATH = MODEL_DIR / "baseline_control_model.pkl"
SEGMENTATION_CSV_PATH = MODEL_DIR / "segmentation_results.csv"
SEGMENT_PLOT_PATH = MODEL_DIR / "segment_distribution.png"


# ---------- 1. LOAD + FILTER + PREPROCESS (must match uplift_models.py exactly) ----------
def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    df = df[df["segment"].isin(["Mens E-Mail", "No E-Mail"])].copy()
    df["treatment"] = (df["segment"] == "Mens E-Mail").astype(int)

    categorical_cols = ["history_segment", "zip_code", "channel"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df = df.reset_index(drop=True)

    exclude_cols = ["segment", "treatment", "visit", "spend", "conversion"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    print("Full dataset shape:", df.shape)
    print("Feature columns:", feature_cols)
    return df, feature_cols


# ---------- 2. LOAD FINAL T-LEARNER, GET UPLIFT SCORES FOR EVERYONE ----------
def get_uplift_scores(df, feature_cols):
    t_learner = joblib.load(T_LEARNER_PATH)
    X = df[feature_cols].values
    uplift = t_learner.predict(X).flatten()
    print(f"\nUplift scores computed for {len(uplift)} customers")
    print("Uplift range:", uplift.min(), "to", uplift.max())
    return uplift


# ---------- 3. TRAIN CONTROL-ONLY BASELINE MODEL ----------
def train_baseline_model(df, feature_cols):
    control_df = df[df["treatment"] == 0]
    X_control = control_df[feature_cols].values
    y_control = control_df["conversion"].values

    baseline_model = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=50,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    baseline_model.fit(X_control, y_control)

    joblib.dump(baseline_model, BASELINE_MODEL_PATH)
    print(f"\nBaseline (control-only) model trained on {len(control_df)} rows")
    print(f"Baseline model saved to: {BASELINE_MODEL_PATH}")
    return baseline_model


# ---------- 4. GET BASELINE CONVERSION PROBABILITY FOR EVERYONE ----------
def get_baseline_probs(baseline_model, df, feature_cols):
    X = df[feature_cols].values
    baseline_prob = baseline_model.predict_proba(X)[:, 1]
    print("Baseline probability range:", baseline_prob.min(), "to", baseline_prob.max())
    return baseline_prob


# ---------- 5. SEGMENT CUSTOMERS (FIXED: uplift-priority quadrant) ----------
def segment_customers(uplift, baseline_prob):
    baseline_median = np.median(baseline_prob)

    # Median of uplift ONLY among non-negative uplift customers
    positive_uplift_mask = uplift >= 0
    uplift_median_positive = np.median(uplift[positive_uplift_mask])

    print(f"\nThresholds used:")
    print(f"  Baseline median (high vs low): {baseline_median:.5f}")
    print(f"  Uplift median among non-negative uplift customers: {uplift_median_positive:.5f}")

    segments = []
    for u, b in zip(uplift, baseline_prob):
        if u < 0:
            segments.append("Sleeping Dogs")
        elif b >= baseline_median and u < uplift_median_positive:
            # high baseline, but NOT high incremental gain -> will convert anyway, low extra value
            segments.append("Sure Things")
        elif u >= uplift_median_positive:
            # high incremental gain -> genuinely worth targeting, regardless of baseline
            segments.append("Persuadables")
        else:
            # low baseline, low uplift
            segments.append("Lost Causes")

    return np.array(segments)


# ---------- 6. SUMMARY ----------
def print_summary(df, uplift, baseline_prob, segments):
    summary_df = pd.DataFrame({
        "segment": segments,
        "uplift": uplift,
        "baseline_prob": baseline_prob
    })

    print("\n" + "="*60)
    print("SEGMENTATION SUMMARY")
    print("="*60)

    counts = summary_df["segment"].value_counts()
    pct = summary_df["segment"].value_counts(normalize=True) * 100

    stats = summary_df.groupby("segment").agg(
        count=("segment", "size"),
        avg_uplift=("uplift", "mean"),
        avg_baseline_prob=("baseline_prob", "mean")
    ).round(4)

    stats["pct_of_customers"] = pct.round(2)
    stats = stats[["count", "pct_of_customers", "avg_uplift", "avg_baseline_prob"]]
    stats = stats.sort_values("avg_uplift", ascending=False)

    print(stats)
    return summary_df


# ---------- 7. PLOT SEGMENT DISTRIBUTION ----------
def plot_segments(summary_df):
    order = ["Persuadables", "Sure Things", "Lost Causes", "Sleeping Dogs"]
    counts = summary_df["segment"].value_counts().reindex(order)

    colors = ["#2ca02c", "#1f77b4", "#7f7f7f", "#d62728"]

    plt.figure(figsize=(8, 5))
    plt.bar(counts.index, counts.values, color=colors)
    plt.ylabel("Number of customers")
    plt.title("Customer Segmentation — Uplift Modeling")
    for i, v in enumerate(counts.values):
        plt.text(i, v + max(counts.values) * 0.01, str(v), ha="center")
    plt.tight_layout()
    plt.savefig(SEGMENT_PLOT_PATH, dpi=150)
    print(f"\nSegment distribution plot saved to: {SEGMENT_PLOT_PATH}")
    plt.close()


# ---------- MAIN PIPELINE ----------
if __name__ == "__main__":
    df, feature_cols = load_and_prepare()

    uplift = get_uplift_scores(df, feature_cols)

    baseline_model = train_baseline_model(df, feature_cols)
    baseline_prob = get_baseline_probs(baseline_model, df, feature_cols)

    segments = segment_customers(uplift, baseline_prob)

    summary_df = print_summary(df, uplift, baseline_prob, segments)

    # Save full results with original (non-preprocessed) columns for readability
    result_df = df.copy()
    result_df["uplift_score"] = uplift
    result_df["baseline_conversion_prob"] = baseline_prob
    result_df["segment"] = segments
    result_df.to_csv(SEGMENTATION_CSV_PATH, index=False)
    print(f"\nFull segmentation results saved to: {SEGMENTATION_CSV_PATH}")

    plot_segments(summary_df)