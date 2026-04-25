from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/processed/score_matchups.parquet")
ARTIFACT_DIR = Path("models/experiments/score/artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["home_pts", "away_pts", "point_spread", "total_points"]
DROP_COLS = ["gameid", "date", "season"]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    print(f"Loaded score_matchups: {df.shape}")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)

    return df


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(model_name: str, target: str, y_true, y_pred) -> dict:
    error = np.abs(y_true - y_pred)

    return {
        "model_name": model_name,
        "target": target,
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "within_5": float(np.mean(error <= 5)),
        "within_10": float(np.mean(error <= 10)),
        "within_15": float(np.mean(error <= 15)),
    }


def make_time_split(df: pd.DataFrame, test_size: float = 0.2):
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    return train_df, test_df


def is_suspicious_feature(col: str) -> bool:
    allowed_if_rolling = "_roll_" in col or "_exp_" in col

    if allowed_if_rolling:
        return False

    leak_terms = [
        "home_pts",
        "away_pts",
        "point_spread",
        "total_points",
        "opp_pts",
        "plus_minus",
    ]

    box_score_suffixes = [
        "_pts",
        "_fgm",
        "_fga",
        "_fgpct",
        "_3pm",
        "_3pa",
        "_3ppct",
        "_ftm",
        "_fta",
        "_ftpct",
        "_oreb",
        "_dreb",
        "_reb",
        "_ast",
        "_tov",
        "_stl",
        "_blk",
        "_pf",
        "_possessions",
        "_off_rating",
        "_def_rating",
        "_net_rating",
        "_ts_proxy",
        "_win",
    ]

    if col in TARGETS:
        return True

    if any(term in col for term in leak_terms):
        return True

    if any(col.endswith(suffix) for suffix in box_score_suffixes):
        return True

    return False


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = set(TARGETS + DROP_COLS)

    features = []
    blocked = []

    for col in df.columns:
        if col in excluded:
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        if is_suspicious_feature(col):
            blocked.append(col)
            continue

        features.append(col)

    if blocked:
        print("\nBlocked suspicious leakage-like columns:")
        for c in blocked:
            print("-", c)

    print(f"\nUsing {len(features)} numeric pre-game features")
    return features


def drop_high_corr_columns(X_train, X_test, threshold: float = 0.95):
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


def build_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(threshold=0.0)),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[("num", numeric_pipeline, feature_cols)],
        remainder="drop",
    )


def fit_ridge(X_train, y_train) -> Pipeline:
    pipe = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(list(X_train.columns))),
            ("model", Ridge(alpha=10.0)),
        ]
    )

    pipe.fit(X_train, y_train)
    return pipe


def fit_gradient_boosting(X_train, y_train) -> Pipeline:
    pipe = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(list(X_train.columns))),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=2,
                    random_state=42,
                ),
            ),
        ]
    )

    pipe.fit(X_train, y_train)
    return pipe


def run_single_target(X_train, X_test, y_train, y_test, target_col: str):
    results = []
    models = {}

    print("\n" + "=" * 60)
    print(f"TARGET: {target_col}")
    print("=" * 60)

    builders = {
        "Ridge": fit_ridge,
        "Gradient Boosting": fit_gradient_boosting,
    }

    for model_name, builder in builders.items():
        model = builder(X_train, y_train)
        preds = model.predict(X_test)

        result = evaluate(model_name, target_col, y_test, preds)
        results.append(result)
        models[model_name] = model

        print(f"\n=== {model_name} ({target_col}) ===")
        print(f"MAE: {result['mae']:.4f}")
        print(f"RMSE: {result['rmse']:.4f}")
        print(f"R²: {result['r2']:.4f}")
        print(f"Within 10: {result['within_10']:.4f}")

    return results, models


def build_prediction_details(test_df, pred_home, pred_away) -> pd.DataFrame:
    keep = [
        c for c in [
            "gameid",
            "date",
            "home_pts",
            "away_pts",
            "point_spread",
            "total_points",
        ]
        if c in test_df.columns
    ]

    details = test_df[keep].copy()
    details["pred_home_pts"] = pred_home
    details["pred_away_pts"] = pred_away
    details["pred_point_spread"] = details["pred_home_pts"] - details["pred_away_pts"]
    details["pred_total_points"] = details["pred_home_pts"] + details["pred_away_pts"]

    details["home_error"] = np.abs(details["home_pts"] - details["pred_home_pts"])
    details["away_error"] = np.abs(details["away_pts"] - details["pred_away_pts"])
    details["spread_error"] = np.abs(details["point_spread"] - details["pred_point_spread"])
    details["total_error"] = np.abs(details["total_points"] - details["pred_total_points"])

    details["spread_direction_correct"] = (
        np.sign(details["point_spread"]) == np.sign(details["pred_point_spread"])
    ).astype(int)

    return details


def print_real_world_metrics(details: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("REAL-WORLD / BETTING-STYLE METRICS")
    print("=" * 60)

    print(f"Spread direction accuracy: {details['spread_direction_correct'].mean():.4f}")
    print(f"Home score within 10: {(details['home_error'] <= 10).mean():.4f}")
    print(f"Away score within 10: {(details['away_error'] <= 10).mean():.4f}")
    print(f"Spread within 10: {(details['spread_error'] <= 10).mean():.4f}")
    print(f"Total within 15: {(details['total_error'] <= 15).mean():.4f}")


def save_model(model, filename: str) -> None:
    path = ARTIFACT_DIR / filename
    joblib.dump(model, path)
    print(f"Saved model: {path}")


def main() -> None:
    df = load_data(DATA_PATH)

    train_df, test_df = make_time_split(df, test_size=0.2)

    feature_cols = get_feature_columns(df)

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    X_train, X_test = drop_high_corr_columns(X_train, X_test, threshold=0.95)

    all_results = []
    best_predictions = {}

    for target_col in TARGETS:
        target_results, target_models = run_single_target(
            X_train=X_train,
            X_test=X_test,
            y_train=train_df[target_col],
            y_test=test_df[target_col],
            target_col=target_col,
        )

        all_results.extend(target_results)

        for model_name, model in target_models.items():
            safe_name = model_name.lower().replace(" ", "_")
            save_model(model, f"{safe_name}_{target_col}.pkl")

        best_result = min(target_results, key=lambda r: r["mae"])
        best_model = target_models[best_result["model_name"]]
        best_predictions[target_col] = best_model.predict(X_test)

    results_df = pd.DataFrame(all_results)
    results_path = ARTIFACT_DIR / "score_results.csv"
    results_df.to_csv(results_path, index=False)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(results_df)

    if "home_pts" in best_predictions and "away_pts" in best_predictions:
        details = build_prediction_details(
            test_df=test_df,
            pred_home=best_predictions["home_pts"],
            pred_away=best_predictions["away_pts"],
        )

        details_path = ARTIFACT_DIR / "score_prediction_details.csv"
        details.to_csv(details_path, index=False)
        print_real_world_metrics(details)

    print("\nSaved:")
    print(f"- {results_path}")
    print(f"- {ARTIFACT_DIR / 'score_prediction_details.csv'}")


if __name__ == "__main__":
    main()