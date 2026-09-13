import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from causalml.inference.meta import BaseTClassifier, BaseXClassifier

# ---------- PATHS ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "hillstrom.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

T_LEARNER_PATH = MODEL_DIR / "t_learner.pkl"
X_LEARNER_PATH = MODEL_DIR / "x_learner.pkl"
PROPENSITY_MODEL_PATH = MODEL_DIR / "propensity_model.pkl"
UPLIFT_RESULTS_PATH = MODEL_DIR / "uplift_results.pkl"
TEST_SPLIT_PATH = MODEL_DIR / "uplift_test_split.pkl"


# ---------- 1. LOAD + FILTER DATA ----------
def load_and_filter():
    df = pd.read_csv(DATA_PATH)

    # Keep only Mens E-Mail (treatment) vs No E-Mail (control) for a clean binary comparison
    df = df[df["segment"].isin(["Mens E-Mail", "No E-Mail"])].copy()
    df["treatment"] = (df["segment"] == "Mens E-Mail").astype(int)

    print("Filtered shape:", df.shape)
    print(df["treatment"].value_counts())
    return df


# ---------- 2. PREPROCESS ----------
def preprocess(df):
    categorical_cols = ["history_segment", "zip_code", "channel"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df


# ---------- 3. BUILD X, treatment, y ----------
def build_xty(df):
    exclude_cols = ["segment", "treatment", "visit", "spend", "conversion"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols]
    treatment = df["treatment"].values
    y = df["conversion"].values

    print("Feature columns:", feature_cols)
    return X, treatment, y, feature_cols


# ---------- 4. TRAIN/TEST SPLIT ----------
def split_data(X, treatment, y):
    X_train, X_test, t_train, t_test, y_train, y_test = train_test_split(
        X, treatment, y, test_size=0.2, random_state=42, stratify=treatment
    )
    print("\nTrain shape:", X_train.shape, "| Test shape:", X_test.shape)

    joblib.dump(
        {"X_test": X_test, "t_test": t_test, "y_test": y_test},
        TEST_SPLIT_PATH
    )
    return X_train, X_test, t_train, t_test, y_train, y_test


# ---------- 5. TRAIN T-LEARNER ----------
def train_t_learner(X_train, t_train, y_train):
    base_rf = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=50,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    t_learner = BaseTClassifier(learner=base_rf)
    t_learner.fit(X=X_train.values, treatment=t_train, y=y_train)

    joblib.dump(t_learner, T_LEARNER_PATH)
    print(f"T-Learner saved to: {T_LEARNER_PATH}")
    return t_learner


# ---------- 6. TRAIN X-LEARNER (with our own propensity model) ----------
def train_x_learner(X_train, t_train, y_train):
    outcome_learner = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=50,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    effect_learner = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=50,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )

    # Train our own propensity model — used for BOTH fit() and predict()
    propensity_model = LogisticRegression(max_iter=2000, random_state=42)
    propensity_model.fit(X_train.values, t_train)
    p_train = propensity_model.predict_proba(X_train.values)[:, 1]

    x_learner = BaseXClassifier(
        outcome_learner=outcome_learner,
        effect_learner=effect_learner
    )
    x_learner.fit(X=X_train.values, treatment=t_train, y=y_train, p=p_train)

    joblib.dump(x_learner, X_LEARNER_PATH)
    joblib.dump(propensity_model, PROPENSITY_MODEL_PATH)
    print(f"X-Learner saved to: {X_LEARNER_PATH}")
    print(f"Propensity model saved to: {PROPENSITY_MODEL_PATH}")
    return x_learner, propensity_model


# ---------- 7. GET UPLIFT SCORES (on TEST set) ----------
def get_uplift_scores(t_learner, x_learner, propensity_model, X_test):
    t_uplift = t_learner.predict(X_test.values).flatten()

    p_test = propensity_model.predict_proba(X_test.values)[:, 1]
    x_uplift = x_learner.predict(X_test.values, p=p_test).flatten()

    results = pd.DataFrame({
        "t_learner_uplift": t_uplift,
        "x_learner_uplift": x_uplift
    })

    print("\nUplift score summary (on held-out TEST set):")
    print(results.describe())

    joblib.dump(results, UPLIFT_RESULTS_PATH)
    print(f"\nUplift results saved to: {UPLIFT_RESULTS_PATH}")
    return results


# ---------- MAIN PIPELINE ----------
if __name__ == "__main__":
    df = load_and_filter()
    df = preprocess(df)
    X, treatment, y, feature_cols = build_xty(df)

    X_train, X_test, t_train, t_test, y_train, y_test = split_data(X, treatment, y)

    t_learner = train_t_learner(X_train, t_train, y_train)
    x_learner, propensity_model = train_x_learner(X_train, t_train, y_train)

    uplift_results = get_uplift_scores(t_learner, x_learner, propensity_model, X_test)