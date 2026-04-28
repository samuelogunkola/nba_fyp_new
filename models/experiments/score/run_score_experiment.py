# allows modern type hint syntax (e.g. list[str] instead of List[str]) on older Python versions
from __future__ import annotations

# Path lets us build file paths in a way that works across operating systems
from pathlib import Path

# joblib saves and loads trained model objects to/from disk
import joblib

# numpy gives us maths tools — used here for RMSE, absolute errors, and correlation matrix work
import numpy as np

# pandas is used to load, sort, and manipulate the matchup dataset
import pandas as pd

# ColumnTransformer applies a pipeline to a specific set of columns and drops everything else
from sklearn.compose import ColumnTransformer

# GradientBoostingRegressor is a tree-based model that builds many small trees in sequence,
# each one correcting the mistakes of the previous — often very accurate but slower to train
from sklearn.ensemble import GradientBoostingRegressor

# VarianceThreshold removes features that barely change across rows (near-constant columns)
# a column that's almost the same value everywhere teaches the model nothing useful
from sklearn.feature_selection import VarianceThreshold

# SimpleImputer fills in missing values using the median of each column
from sklearn.impute import SimpleImputer

# Ridge is a linear regression model with L2 regularisation to prevent overfitting
from sklearn.linear_model import Ridge

# standard regression evaluation metrics from scikit-learn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Pipeline chains multiple preprocessing and modelling steps into a single object
from sklearn.pipeline import Pipeline

# StandardScaler rescales features to zero mean and unit variance — important for Ridge
from sklearn.preprocessing import StandardScaler


# path to the processed score matchup dataset (parquet is faster to load than CSV)
DATA_PATH = Path("data/processed/score_matchups.parquet")

# folder where all trained models and result CSVs will be written
ARTIFACT_DIR = Path("models/experiments/score/artifacts")

# create the folder if it doesn't exist (parents=True handles any missing parent folders too)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# the four things we want to predict — each gets its own separate model
TARGETS = ["home_pts", "away_pts", "point_spread", "total_points"]

# columns we always drop before building features — these are identifiers, not signals
DROP_COLS = ["gameid", "date", "season"]


def load_data(path: Path) -> pd.DataFrame:
    # support both parquet and CSV formats — check the file extension to decide which to use
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    print(f"Loaded score_matchups: {df.shape}")

    if "date" in df.columns:
        # parse dates so we can sort rows chronologically
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        # sort oldest first so the time-based train/test split is always chronologically correct
        df = df.sort_values("date").reset_index(drop=True)

    return df


def rmse(y_true, y_pred) -> float:
    # RMSE (root mean squared error) — square errors, average them, then take the square root
    # squaring means big prediction misses are punished much more than small ones
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(model_name: str, target: str, y_true, y_pred) -> dict:
    # compute the absolute error for each prediction (direction doesn't matter, just magnitude)
    error = np.abs(y_true - y_pred)

    return {
        "model_name": model_name,
        "target": target,
        "mae": mean_absolute_error(y_true, y_pred),  # average error in points
        "rmse": rmse(y_true, y_pred),                # penalises large errors more than MAE
        "r2": r2_score(y_true, y_pred),              # how much variance the model explains (0–1)
        # practical accuracy bands — what fraction of predictions landed within N points of reality?
        # these are more meaningful to a sports fan than abstract statistical metrics
        "within_5": float(np.mean(error <= 5)),
        "within_10": float(np.mean(error <= 10)),
        "within_15": float(np.mean(error <= 15)),
    }


def make_time_split(df: pd.DataFrame, test_size: float = 0.2):
    # split the dataframe chronologically: first 80% for training, last 20% for testing
    # because the data is sorted by date, this ensures the model only ever trains on past games
    # and is tested on games it has never seen — just like real-world prediction
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    return train_df, test_df


def is_suspicious_feature(col: str) -> bool:
    # decides whether a column looks like it could be leaking post-game information
    # data leakage is when the model accidentally sees the answer before making a prediction

    # rolling and exponential moving average features are safe — they only look backwards
    # so if a column name contains "_roll_" or "_exp_", we trust it and allow it through
    allowed_if_rolling = "_roll_" in col or "_exp_" in col

    if allowed_if_rolling:
        return False

    # these strings in a column name strongly suggest it contains same-game results
    # e.g. "home_pts" is the actual final score — the model can't know this before tip-off
    leak_terms = [
        "home_pts",
        "away_pts",
        "point_spread",
        "total_points",
        "opp_pts",
        "plus_minus",
    ]

    # these suffixes typically indicate raw box-score stats from the same game
    # e.g. "home_fgm" = field goals made in this game — only available after the game ends
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

    # block the column if it's one of the prediction targets
    if col in TARGETS:
        return True

    # block if the column name contains any of the known leakage terms
    if any(term in col for term in leak_terms):
        return True

    # block if the column ends with any of the raw box-score suffixes
    if any(col.endswith(suffix) for suffix in box_score_suffixes):
        return True

    # if none of the above matched, the column looks safe to use
    return False


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    # build the final list of feature columns by filtering out anything unsafe or non-numeric
    excluded = set(TARGETS + DROP_COLS)

    features = []  # columns we'll actually use
    blocked = []   # columns we filtered out due to suspected leakage

    for col in df.columns:
        # skip identifiers and target columns
        if col in excluded:
            continue

        # skip non-numeric columns — models can't work with strings or dates directly
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        # run the leakage check — move suspicious columns to the blocked list
        if is_suspicious_feature(col):
            blocked.append(col)
            continue

        features.append(col)

    # print the blocked columns so we have a clear audit trail of what was removed
    if blocked:
        print("\nBlocked suspicious leakage-like columns:")
        for c in blocked:
            print("-", c)

    print(f"\nUsing {len(features)} numeric pre-game features")
    return features


def drop_high_corr_columns(X_train, X_test, threshold: float = 0.95):
    # remove features that are extremely highly correlated with each other (above 95% by default)
    # keeping two near-identical features doesn't add information and can slow training down

    # compute the correlation matrix for all feature pairs in the training set
    corr = X_train.corr().abs()

    # np.triu extracts the upper triangle of the matrix (above the diagonal)
    # this avoids counting each pair twice (A-B and B-A are the same pair)
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    # find any column where at least one other column has a correlation above the threshold
    to_drop = [
        col for col in upper.columns
        if any(upper[col] > threshold)
    ]

    # drop those columns from both sets so they always have matching shapes
    X_train = X_train.drop(columns=to_drop, errors="ignore")
    X_test = X_test.drop(columns=to_drop, errors="ignore")

    print(f"Dropped {len(to_drop)} highly correlated features")
    print(f"Remaining features: {X_train.shape[1]}")

    return X_train, X_test


def build_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    # build the preprocessing pipeline that gets applied before every model
    # chaining these steps into a pipeline guarantees they always run in the same order
    numeric_pipeline = Pipeline(
        steps=[
            # fill any missing values with the median of each column (learned from training set)
            ("imputer", SimpleImputer(strategy="median")),
            # drop any columns with zero variance — same value in every row is useless
            ("variance", VarianceThreshold(threshold=0.0)),
            # rescale all features to zero mean and unit variance — important for Ridge
            ("scaler", StandardScaler()),
        ]
    )

    # ColumnTransformer applies the pipeline to the specified columns
    # remainder="drop" discards any columns not in feature_cols rather than passing them through
    return ColumnTransformer(
        transformers=[("num", numeric_pipeline, feature_cols)],
        remainder="drop",
    )


def fit_ridge(X_train, y_train) -> Pipeline:
    # build and train a Ridge regression pipeline
    # Ridge adds a penalty for large coefficients, which reduces overfitting on noisy data
    pipe = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(list(X_train.columns))),
            ("model", Ridge(alpha=10.0)),  # alpha=10 means fairly strong regularisation
        ]
    )

    pipe.fit(X_train, y_train)
    return pipe


def fit_gradient_boosting(X_train, y_train) -> Pipeline:
    # build and train a Gradient Boosting regression pipeline
    # this builds many shallow trees, each one correcting the previous tree's errors
    pipe = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(list(X_train.columns))),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=100,    # number of trees to build in the sequence
                    learning_rate=0.05, # how much each new tree corrects the previous ones
                    max_depth=2,        # keep trees shallow to prevent overfitting
                    random_state=42,    # fixed seed so results are reproducible
                ),
            ),
        ]
    )

    pipe.fit(X_train, y_train)
    return pipe


def run_single_target(X_train, X_test, y_train, y_test, target_col: str):
    # trains and evaluates both models (Ridge and Gradient Boosting) for one prediction target
    # returns all results and all trained model objects for this target
    results = []
    models = {}

    print("\n" + "=" * 60)
    print(f"TARGET: {target_col}")
    print("=" * 60)

    # dictionary mapping model names to their training functions
    # makes it easy to loop over models without duplicating code
    builders = {
        "Ridge": fit_ridge,
        "Gradient Boosting": fit_gradient_boosting,
    }

    for model_name, builder in builders.items():
        # train the model and generate test set predictions
        model = builder(X_train, y_train)
        preds = model.predict(X_test)

        # evaluate and record the results
        result = evaluate(model_name, target_col, y_test, preds)
        results.append(result)
        models[model_name] = model

        # print a quick summary for this model/target combination
        print(f"\n=== {model_name} ({target_col}) ===")
        print(f"MAE: {result['mae']:.4f}")
        print(f"RMSE: {result['rmse']:.4f}")
        print(f"R²: {result['r2']:.4f}")
        print(f"Within 10: {result['within_10']:.4f}")

    return results, models


def build_prediction_details(test_df, pred_home, pred_away) -> pd.DataFrame:
    # build a detailed per-game results dataframe that shows predictions alongside actual scores
    # this is what gets saved to CSV and displayed in the dashboard's score explorer

    # only include columns that actually exist in the test dataframe
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

    # attach the home and away score predictions
    details["pred_home_pts"] = pred_home
    details["pred_away_pts"] = pred_away

    # derive spread and total from the home and away predictions
    # spread = home score minus away score (positive means home team winning)
    details["pred_point_spread"] = details["pred_home_pts"] - details["pred_away_pts"]

    # total = combined score of both teams (the "over/under" in betting)
    details["pred_total_points"] = details["pred_home_pts"] + details["pred_away_pts"]

    # compute absolute errors for each of the four predictions
    details["home_error"] = np.abs(details["home_pts"] - details["pred_home_pts"])
    details["away_error"] = np.abs(details["away_pts"] - details["pred_away_pts"])
    details["spread_error"] = np.abs(details["point_spread"] - details["pred_point_spread"])
    details["total_error"] = np.abs(details["total_points"] - details["pred_total_points"])

    # check whether the model got the spread direction right (did it correctly pick which team would lead?)
    # np.sign returns -1, 0, or 1 depending on the sign of the value
    # if both the real and predicted spreads have the same sign, the direction was correct
    details["spread_direction_correct"] = (
        np.sign(details["point_spread"]) == np.sign(details["pred_point_spread"])
    ).astype(int)

    return details


def print_real_world_metrics(details: pd.DataFrame) -> None:
    # print practical "real-world" style metrics that go beyond standard MAE/R²
    # these are the kind of numbers a sports analyst or bettor would actually care about
    print("\n" + "=" * 60)
    print("REAL-WORLD / BETTING-STYLE METRICS")
    print("=" * 60)

    # what fraction of games did the model correctly predict which team would score more?
    print(f"Spread direction accuracy: {details['spread_direction_correct'].mean():.4f}")

    # what fraction of home/away score predictions were within 10 points of the real score?
    print(f"Home score within 10: {(details['home_error'] <= 10).mean():.4f}")
    print(f"Away score within 10: {(details['away_error'] <= 10).mean():.4f}")

    # how often was the predicted spread within 10 points of the real spread?
    print(f"Spread within 10: {(details['spread_error'] <= 10).mean():.4f}")

    # how often was the predicted total within 15 points of the real combined score?
    print(f"Total within 15: {(details['total_error'] <= 15).mean():.4f}")


def save_model(model, filename: str) -> None:
    # save a trained model to the artifact directory using joblib
    path = ARTIFACT_DIR / filename
    joblib.dump(model, path)
    print(f"Saved model: {path}")


def main() -> None:
    # load and sort the full score matchup dataset
    df = load_data(DATA_PATH)

    # split into training (80%) and test (20%) sets chronologically
    train_df, test_df = make_time_split(df, test_size=0.2)

    # get the safe, leakage-free feature columns
    feature_cols = get_feature_columns(df)

    # extract the feature matrices and replace any infinity values with NaN
    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    # remove features that are extremely correlated with each other to reduce redundancy
    X_train, X_test = drop_high_corr_columns(X_train, X_test, threshold=0.95)

    # we'll collect results from every model/target combination here
    all_results = []

    # store the best model's predictions for each target so we can build the details CSV later
    best_predictions = {}

    # run experiments for each of the four prediction targets in turn
    for target_col in TARGETS:
        target_results, target_models = run_single_target(
            X_train=X_train,
            X_test=X_test,
            y_train=train_df[target_col],
            y_test=test_df[target_col],
            target_col=target_col,
        )

        # add this target's results to the master list
        all_results.extend(target_results)

        # save every model variant (Ridge and Gradient Boosting) for this target
        for model_name, model in target_models.items():
            # convert "Gradient Boosting" to "gradient_boosting" for a clean filename
            safe_name = model_name.lower().replace(" ", "_")
            save_model(model, f"{safe_name}_{target_col}.pkl")

        # find the best-performing model for this target (by MAE) to use in the details CSV
        best_result = min(target_results, key=lambda r: r["mae"])
        best_model = target_models[best_result["model_name"]]
        best_predictions[target_col] = best_model.predict(X_test)

    # save all results to a single CSV — one row per model/target combination
    results_df = pd.DataFrame(all_results)
    results_path = ARTIFACT_DIR / "score_results.csv"
    results_df.to_csv(results_path, index=False)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(results_df)

    # build and save the per-game prediction details CSV if we have home and away predictions
    if "home_pts" in best_predictions and "away_pts" in best_predictions:
        details = build_prediction_details(
            test_df=test_df,
            pred_home=best_predictions["home_pts"],
            pred_away=best_predictions["away_pts"],
        )

        details_path = ARTIFACT_DIR / "score_prediction_details.csv"
        details.to_csv(details_path, index=False)

        # print the practical real-world metrics to the console
        print_real_world_metrics(details)

    # confirm where output files were written
    print("\nSaved:")
    print(f"- {results_path}")
    print(f"- {ARTIFACT_DIR / 'score_prediction_details.csv'}")


# only run main() when this script is executed directly from the command line
# if it's imported as a module by another script, main() won't run automatically
if __name__ == "__main__":
    main()
