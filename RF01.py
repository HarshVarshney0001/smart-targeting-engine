import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------- PATHS ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "hillstrom.csv"
MODEL_PATH = BASE_DIR / "models" / "rf_baseline.pkl"

MODEL_PATH.parent.mkdir(exist_ok=True)  # models folder na ho toh bana dega


# ---------- 1. EDA ----------
def run_eda(df):
    print("="*50)
    print("EDA")
    print("="*50)
    print("Shape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nConversion class balance:\n", df["conversion"].value_counts(normalize=True))
    print("\nSegment (treatment) distribution:\n", df["segment"].value_counts())
    print("\nFirst 5 rows:\n", df.head())


# ---------- 2. PREPROCESSING ----------
def preprocess(df):
    df = df.dropna()

    categorical_cols = ["history_segment", "zip_code", "channel"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    print("\nAfter preprocessing, shape:", df.shape)
    return df


# ---------- 3. FEATURE ENGINEERING ----------
def build_features(df):
    # visit, spend, segment, conversion are OUTCOMES -> leakage if used as features
    exclude_cols = ["segment", "visit", "spend", "conversion"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols]
    y = df["conversion"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nFeature columns used:", feature_cols)
    print("X_train shape:", X_train.shape, "| X_test shape:", X_test.shape)
    return X_train, X_test, y_train, y_test


# ---------- 4. TRAIN RANDOM FOREST ----------
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    print(f"\nModel trained and saved to: {MODEL_PATH}")
    return model


# ---------- 5. PREDICTION ----------
def predict(model, X_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return y_pred, y_proba


# ---------- 6. EVALUATION ----------
def evaluate(y_test, y_pred):
    print("\n" + "="*50)
    print("EVALUATION")
    print("="*50)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


# ---------- MAIN PIPELINE ----------
if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    run_eda(df)
    df_processed = preprocess(df)
    X_train, X_test, y_train, y_test = build_features(df_processed)
    model = train_model(X_train, y_train)
    y_pred, y_proba = predict(model, X_test)
    evaluate(y_test, y_pred)