from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold


DATA_PATH = Path("data/processed/player_stats_dataset.parquet")
ARTIFACT_DIR = Path("models/experiments/player/artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["pts", "reb", "ast"]

DROP_COLS = [
    "gameid", "date", "playerid", "player",
    "team", "home", "away", "position",
    "pts", "reb", "ast",
]


def load_data():
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded player dataset: {df.shape}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate(name, target, y_true, y_pred):
    err = np.abs(y_true - y_pred)
    return {
        "model": name,
        "target": target,
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "within_2": float(np.mean(err <= 2)),
        "within_5": float(np.mean(err <= 5)),
    }


def get_features(df):
    features = []

    for col in df.columns:
        if col in DROP_COLS:
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        # Keep only pre-game rolling features and safe context
        if "_roll_" in col or col in ["min"]:
            features.append(col)

    print(f"Using {len(features)} fast pre-game features")
    return features


def build_model(cols):
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("var", VarianceThreshold()),
                ("scale", StandardScaler()),
            ]), cols)
        ])),
        ("model", Ridge(alpha=10.0))
    ])


def main():
    df = load_data()

    # Fast sample for experimentation
    # Still large enough for strong results
    df = df.tail(200_000).copy()
    print(f"Using recent sample: {df.shape}")

    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    print(f"Train: {train.shape}")
    print(f"Test: {test.shape}")

    features = get_features(df)

    X_train = train[features].replace([np.inf, -np.inf], np.nan)
    X_test = test[features].replace([np.inf, -np.inf], np.nan)

    results = []
    predictions = test[[c for c in ["gameid", "date", "player", "team", "pts", "reb", "ast"] if c in test.columns]].copy()

    for target in TARGETS:
        print("\n" + "=" * 60)
        print(f"TARGET: {target}")
        print("=" * 60)

        model = build_model(features)
        model.fit(X_train, train[target])

        preds = model.predict(X_test)

        result = evaluate("Ridge", target, test[target], preds)
        results.append(result)

        print(f"MAE: {result['mae']:.4f}")
        print(f"RMSE: {result['rmse']:.4f}")
        print(f"R²: {result['r2']:.4f}")
        print(f"Within 2: {result['within_2']:.4f}")
        print(f"Within 5: {result['within_5']:.4f}")

        predictions[f"pred_{target}"] = preds
        joblib.dump(model, ARTIFACT_DIR / f"ridge_{target}.pkl")

    results_df = pd.DataFrame(results)
    results_df.to_csv(ARTIFACT_DIR / "player_results_fast.csv", index=False)
    predictions.to_csv(ARTIFACT_DIR / "player_prediction_details_fast.csv", index=False)

    print("\nFINAL RESULTS")
    print(results_df)

    print("\nSaved:")
    print(f"- {ARTIFACT_DIR / 'player_results_fast.csv'}")
    print(f"- {ARTIFACT_DIR / 'player_prediction_details_fast.csv'}")


if __name__ == "__main__":
    main()