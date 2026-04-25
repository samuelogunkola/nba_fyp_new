from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold


DATA_PATH = Path("data/processed/win_matchups.parquet")
ARTIFACT_DIR = Path("models/experiments/win/artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "home_win"
DROP_COLS = ["gameid", "date", "season", TARGET]


def load_data(path):
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    print(f"Loaded win_matchups: {df.shape}")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)

    return df


def is_leakage_col(col):
    allowed_if_rolling = "_roll_" in col or "_exp_" in col or col.startswith("diff_")

    if allowed_if_rolling:
        return False

    leakage_terms = [
        "home_win",
        "away_win",
        "winner",
        "result",
        "home_pts",
        "away_pts",
        "point_spread",
        "total_points",
        "plus_minus",
    ]

    if col == TARGET:
        return True

    return any(term in col for term in leakage_terms)


def get_feature_columns(df):
    features = []
    blocked = []

    for col in df.columns:
        if col in DROP_COLS:
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        if is_leakage_col(col):
            blocked.append(col)
            continue

        features.append(col)

    if blocked:
        print("\nBlocked possible leakage columns:")
        for col in blocked:
            print("-", col)

    print(f"\nUsing {len(features)} numeric pre-game features")
    return features


def make_time_split(df, test_size=0.2):
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    return train_df, test_df


def drop_high_corr_columns(X_train, X_test, threshold=0.95):
    corr = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = [
        col for col in upper.columns
        if any(upper[col] > threshold)
    ]

    X_train = X_train.drop(columns=to_drop, errors="ignore")
    X_test = X_test.drop(columns=to_drop, errors="ignore")

    print(f"Dropped {len(to_drop)} highly correlated features")
    print(f"Remaining features: {X_train.shape[1]}")

    return X_train, X_test


def build_preprocessor(feature_cols):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold()),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[("num", numeric_pipeline, feature_cols)],
        remainder="drop",
    )


def evaluate_classifier(model_name, y_true, y_pred, y_prob):
    result = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }

    print(f"\n=== {model_name} ===")
    print(f"Accuracy:  {result['accuracy']:.4f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall:    {result['recall']:.4f}")
    print(f"F1:        {result['f1']:.4f}")
    print(f"ROC-AUC:   {result['roc_auc']:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    return result


def get_models(feature_cols):
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(feature_cols)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),

        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(feature_cols)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        max_depth=10,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        "Gradient Boosting": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(feature_cols)),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=150,
                        learning_rate=0.05,
                        max_depth=2,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def save_model(model, filename):
    path = ARTIFACT_DIR / filename
    joblib.dump(model, path)
    print(f"Saved model: {path}")


def main():
    df = load_data(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found. Available columns include: {df.columns[:30].tolist()}")

    train_df, test_df = make_time_split(df)

    feature_cols = get_feature_columns(df)

    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan)

    y_train = train_df[TARGET].astype(int)
    y_test = test_df[TARGET].astype(int)

    X_train, X_test = drop_high_corr_columns(X_train, X_test)

    feature_cols = list(X_train.columns)

    models = get_models(feature_cols)

    results = []
    predictions = test_df[[c for c in ["gameid", "date", "season", TARGET] if c in test_df.columns]].copy()

    best_model_name = None
    best_f1 = -1

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred

        result = evaluate_classifier(model_name, y_test, y_pred, y_prob)
        results.append(result)

        safe_name = model_name.lower().replace(" ", "_")
        save_model(model, f"{safe_name}_home_win.pkl")

        predictions[f"{safe_name}_pred"] = y_pred
        predictions[f"{safe_name}_prob"] = y_prob

        if result["f1"] > best_f1:
            best_f1 = result["f1"]
            best_model_name = model_name

    results_df = pd.DataFrame(results)
    results_path = ARTIFACT_DIR / "win_results.csv"
    predictions_path = ARTIFACT_DIR / "win_prediction_details.csv"

    results_df.to_csv(results_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    print("\n" + "=" * 60)
    print("WIN MODEL RESULTS SUMMARY")
    print("=" * 60)
    print(results_df)

    print(f"\nBest model by F1: {best_model_name} ({best_f1:.4f})")

    print("\nSaved:")
    print(f"- {results_path}")
    print(f"- {predictions_path}")


if __name__ == "__main__":
    main()