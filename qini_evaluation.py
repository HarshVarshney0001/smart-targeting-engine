import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from causalml.inference.meta import BaseTClassifier, BaseXClassifier

# ---------- PATHS ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "hillstrom.csv"
MODEL_DIR = BASE_DIR / "models"
QINI_PLOT_PATH = MODEL_DIR / "qini_curve.png"
QINI_SCORES_PATH = MODEL_DIR / "qini_scores.pkl"

N_FOLDS = 5
RANDOM_STATE = 42


# ---------- 1. LOAD + FILTER + PREPROCESS (same logic as uplift_models.py) ----------
def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    df = df[df["segment"].isin(["Mens E-Mail", "No E-Mail"])].copy()
    df["treatment"] = (df["segment"] == "Mens E-Mail").astype(int)

    categorical_cols = ["history_segment", "zip_code", "channel"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    df = df.reset_index(drop=True)  # ensures clean 0..n-1 positional indices

    exclude_cols = ["segment", "treatment", "visit", "spend", "conversion"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].values
    treatment = df["treatment"].values
    y = df["conversion"].values

    print("Full dataset shape:", X.shape)
    print("Feature columns:", feature_cols)
    return X, treatment, y


# ---------- 2. CROSS-VALIDATED OUT-OF-FOLD UPLIFT SCORING ----------
def run_cv_uplift(X, treatment, y):
    n = len(y)
    t_uplift_oof = np.zeros(n)
    x_uplift_oof = np.zeros(n)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_num = 1
    for train_idx, test_idx in skf.split(X, treatment):
        print(f"\n--- Fold {fold_num}/{N_FOLDS} ---")
        X_train, X_test = X[train_idx], X[test_idx]
        t_train, t_test = treatment[train_idx], treatment[test_idx]
        y_train = y[train_idx]

        # ---- T-Learner: train on this fold's train split, predict on held-out fold ----
        t_base_rf = RandomForestClassifier(
            n_estimators=200, min_samples_leaf=50, max_depth=8,
            random_state=RANDOM_STATE, n_jobs=-1
        )
        t_learner = BaseTClassifier(learner=t_base_rf)
        t_learner.fit(X=X_train, treatment=t_train, y=y_train)
        t_uplift_oof[test_idx] = t_learner.predict(X_test).flatten()

        # ---- X-Learner: same fold split, with its own propensity model ----
        outcome_learner = RandomForestClassifier(
            n_estimators=200, min_samples_leaf=50, max_depth=8,
            random_state=RANDOM_STATE, n_jobs=-1
        )
        effect_learner = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=50, max_depth=8,
            random_state=RANDOM_STATE, n_jobs=-1
        )
        propensity_model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        propensity_model.fit(X_train, t_train)
        p_train = propensity_model.predict_proba(X_train)[:, 1]

        x_learner = BaseXClassifier(outcome_learner=outcome_learner, effect_learner=effect_learner)
        x_learner.fit(X=X_train, treatment=t_train, y=y_train, p=p_train)

        p_test = propensity_model.predict_proba(X_test)[:, 1]
        x_uplift_oof[test_idx] = x_learner.predict(X_test, p=p_test).flatten()

        fold_num += 1

    print("\nAll folds complete. Out-of-fold predictions collected for full dataset.")
    return t_uplift_oof, x_uplift_oof


# ---------- 3. BUILD QINI CURVE FOR ONE MODEL ----------
def build_qini_curve(df, score_col):
    data = df.sort_values(score_col, ascending=False).reset_index(drop=True)

    data["cum_treat_count"] = (data["treatment"] == 1).cumsum()
    data["cum_ctrl_count"] = (data["treatment"] == 0).cumsum()
    data["cum_treat_conv"] = (data["conversion"] * (data["treatment"] == 1)).cumsum()
    data["cum_ctrl_conv"] = (data["conversion"] * (data["treatment"] == 0)).cumsum()

    ratio = data["cum_treat_count"] / data["cum_ctrl_count"].replace(0, np.nan)
    data["qini_gain"] = data["cum_treat_conv"] - data["cum_ctrl_conv"] * ratio
    data["qini_gain"] = data["qini_gain"].fillna(0)

    n = len(data)
    data["population_fraction"] = np.arange(1, n + 1) / n

    zero_row = pd.DataFrame({"population_fraction": [0], "qini_gain": [0]})
    curve = pd.concat([zero_row, data[["population_fraction", "qini_gain"]]], ignore_index=True)
    return curve


# ---------- 4. QINI COEFFICIENT ----------
def qini_coefficient(curve):
    x = curve["population_fraction"].values
    y = curve["qini_gain"].values

    model_area = np.sum((y[1:] + y[:-1]) / 2 * np.diff(x))
    final_gain = y[-1]
    random_area = 0.5 * final_gain

    return model_area - random_area


# ---------- 5. PLOT ----------
def plot_qini(curve_t, curve_x, coef_t, coef_x):
    plt.figure(figsize=(9, 6))

    plt.plot(curve_t["population_fraction"], curve_t["qini_gain"],
              label=f"T-Learner (Qini={coef_t:.3f})", linewidth=2)
    plt.plot(curve_x["population_fraction"], curve_x["qini_gain"],
              label=f"X-Learner (Qini={coef_x:.3f})", linewidth=2)

    final_gain = curve_t["qini_gain"].iloc[-1]
    plt.plot([0, 1], [0, final_gain], linestyle="--", color="gray", label="Random targeting")

    plt.xlabel("Fraction of customers targeted (sorted by uplift score)")
    plt.ylabel("Cumulative incremental conversions")
    plt.title("Qini Curve — T-Learner vs X-Learner (5-fold CV, out-of-fold)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(QINI_PLOT_PATH, dpi=150)
    print(f"\nQini curve plot saved to: {QINI_PLOT_PATH}")
    plt.close()


# ---------- MAIN PIPELINE ----------
if __name__ == "__main__":
    X, treatment, y = load_and_prepare()

    t_uplift_oof, x_uplift_oof = run_cv_uplift(X, treatment, y)

    df_eval = pd.DataFrame({
        "treatment": treatment,
        "conversion": y,
        "t_learner_uplift": t_uplift_oof,
        "x_learner_uplift": x_uplift_oof,
    })
    print("\nOut-of-fold evaluation data shape:", df_eval.shape)
    print("Total conversions available for Qini evaluation:", df_eval["conversion"].sum())

    curve_t = build_qini_curve(df_eval, "t_learner_uplift")
    curve_x = build_qini_curve(df_eval, "x_learner_uplift")

    coef_t = qini_coefficient(curve_t)
    coef_x = qini_coefficient(curve_x)

    print("\n" + "="*50)
    print("QINI COEFFICIENTS (5-fold CV, out-of-fold)")
    print("="*50)
    print(f"T-Learner Qini coefficient: {coef_t:.4f}")
    print(f"X-Learner Qini coefficient: {coef_x:.4f}")

    best_model = "T-Learner" if coef_t > coef_x else "X-Learner"
    print(f"\nBest model: {best_model}")

    plot_qini(curve_t, curve_x, coef_t, coef_x)

    joblib.dump(
        {"t_learner_qini": coef_t, "x_learner_qini": coef_x, "best_model": best_model},
        QINI_SCORES_PATH
    )
    print(f"Qini scores saved to: {QINI_SCORES_PATH}")