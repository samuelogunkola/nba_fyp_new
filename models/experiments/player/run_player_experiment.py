# Path lets us build file paths that work across different operating systems
from pathlib import Path

# joblib is used to save trained models to disk so they can be loaded later without retraining
import joblib

# numpy gives us maths tools — used here for RMSE calculation and checking prediction accuracy
import numpy as np

# pandas is used to load, sort, and manipulate the player dataset
import pandas as pd

# ColumnTransformer lets us apply a preprocessing pipeline to a specific subset of columns
from sklearn.compose import ColumnTransformer

# SimpleImputer fills in missing values — we use the median of each column
from sklearn.impute import SimpleImputer

# Ridge is a linear regression model with L2 regularisation
# regularisation helps prevent overfitting by penalising very large coefficients
from sklearn.linear_model import Ridge

# standard regression evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Pipeline chains multiple steps (imputing, scaling, modelling) into one object
# this ensures the same transformations are always applied in the same order
from sklearn.pipeline import Pipeline

# StandardScaler rescales each feature to have zero mean and unit variance
# Ridge regression benefits from scaled features because it's sensitive to feature magnitude
from sklearn.preprocessing import StandardScaler

# VarianceThreshold removes features that barely change across rows
# a feature with near-zero variance carries no useful signal for the model
from sklearn.feature_selection import VarianceThreshold


# path to the full processed player dataset (stored as parquet for faster loading than CSV)
DATA_PATH = Path("data/processed/player_stats_dataset.parquet")

# folder where all output files from this experiment will be saved
ARTIFACT_DIR = Path("models/experiments/player/artifacts")

# create the folder if it doesn't already exist (parents=True creates any missing parent folders)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# the three player stats we want to train a separate model for
TARGETS = ["pts", "reb", "ast"]

# columns that must never be used as model inputs
# these are either identifiers (gameid, player) or the actual outcomes we're predicting
# including the targets as features would be pure data leakage
DROP_COLS = [
    "gameid", "date", "playerid", "player",
    "team", "home", "away", "position",
    "pts",   # actual points — would be leakage
    "reb",   # actual rebounds — would be leakage
    "ast",   # actual assists — would be leakage
]


def load_data():
    # read the parquet file into a dataframe — parquet is much faster than CSV for large files
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded player dataset: {df.shape}")

    # parse the date column so we can sort rows chronologically
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # sort oldest to newest so the 80/20 split later gives us a proper time-based train/test split
    df = df.sort_values("date").reset_index(drop=True)
    return df


def rmse(y_true, y_pred):
    # RMSE (root mean squared error) — square each error, average them, then take the square root
    # squaring means large mistakes are penalised more heavily than small ones
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate(name, target, y_true, y_pred):
    # compute the absolute error for each prediction (ignoring whether we were over or under)
    err = np.abs(y_true - y_pred)

    return {
        "model": name,
        "target": target,
        "mae": mean_absolute_error(y_true, y_pred),  # average error in the same units as the stat
        "rmse": rmse(y_true, y_pred),                # penalises big misses more than MAE does
        "r2": r2_score(y_true, y_pred),              # how much variance the model explains (0–1)
        # practical metrics: what fraction of predictions were within 2 or 5 of the real value?
        # e.g. within_2 of 0.35 means 35% of point predictions landed within ±2 of actual
        "within_2": float(np.mean(err <= 2)),
        "within_5": float(np.mean(err <= 5)),
    }


def get_features(df):
    # build the list of columns the model should actually use as inputs
    features = []

    for col in df.columns:
        # skip anything in the drop list — identifiers and target columns
        if col in DROP_COLS:
            continue

        # skip non-numeric columns — models can't learn from raw strings
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        # only keep rolling average features and the minutes column
        # rolling features capture recent form without leaking same-game results
        # minutes played is a direct proxy for a player's current role and workload
        if "_roll_" in col or col in ["min"]:
            features.append(col)

    print(f"Using {len(features)} fast pre-game features")
    return features


def build_model(cols):
    # build a scikit-learn Pipeline that chains preprocessing and the Ridge model together
    # using a Pipeline means we can call model.fit() and model.predict() cleanly,
    # and the preprocessing steps are automatically applied at both training and inference time
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", Pipeline([
                # step 1: fill any missing values with the median of each column
                ("impute", SimpleImputer(strategy="median")),
                # step 2: drop any features whose values barely vary at all
                # a feature that is almost always the same value teaches the model nothing
                ("var", VarianceThreshold()),
                # step 3: rescale all features to have mean 0 and standard deviation 1
                # Ridge regression works best when features are on comparable scales
                ("scale", StandardScaler()),
            ]), cols)
        ])),
        # the actual model: Ridge regression with alpha=10 for regularisation strength
        # higher alpha = more regularisation = coefficients are kept smaller = less overfitting
        ("model", Ridge(alpha=10.0))
    ])


def main():
    # load and sort the full player dataset
    df = load_data()

    # take only the most recent 200,000 rows to keep training time manageable
    # these are the most recent games so the patterns are more relevant to current performance
    # still large enough for strong results
    df = df.tail(200_000).copy()
    print(f"Using recent sample: {df.shape}")

    # use an 80/20 split: first 80% of rows for training, last 20% for testing
    # because the data is sorted by date this is automatically a time-based split —
    # we always train on older games and test on newer ones
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    print(f"Train: {train.shape}")
    print(f"Test: {test.shape}")

    # get the list of valid input features from the full dataframe
    features = get_features(df)

    # extract the feature columns from the train and test sets
    # replace any infinity values with NaN so the imputer can handle them
    X_train = train[features].replace([np.inf, -np.inf], np.nan)
    X_test = test[features].replace([np.inf, -np.inf], np.nan)

    # list to collect one results dictionary per target
    results = []

    # also keep a copy of the test rows with identifiers so we can save predictions alongside actuals
    predictions = test[[c for c in ["gameid", "date", "player", "team", "pts", "reb", "ast"] if c in test.columns]].copy()

    # train a separate Ridge model for each of the three targets
    for target in TARGETS:
        print("\n" + "=" * 60)
        print(f"TARGET: {target}")
        print("=" * 60)

        # build a fresh pipeline for each target — each model is independent
        model = build_model(features)

        # train the model on the training set for this specific target
        model.fit(X_train, train[target])

        # generate predictions on the unseen test set
        preds = model.predict(X_test)

        # compute all evaluation metrics and add them to the results list
        result = evaluate("Ridge", target, test[target], preds)
        results.append(result)

        # print a summary for this target to the console
        print(f"MAE: {result['mae']:.4f}")
        print(f"RMSE: {result['rmse']:.4f}")
        print(f"R²: {result['r2']:.4f}")
        print(f"Within 2: {result['within_2']:.4f}")
        print(f"Within 5: {result['within_5']:.4f}")

        # attach this model's predictions as a new column in the predictions dataframe
        predictions[f"pred_{target}"] = preds

        # save the trained model to disk so the dashboard can load it for inference later
        joblib.dump(model, ARTIFACT_DIR / f"ridge_{target}.pkl")

    # combine all per-target result dictionaries into a single dataframe and save as CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(ARTIFACT_DIR / "player_results_fast.csv", index=False)

    # save the per-game predictions alongside the real values for inspection and debugging
    predictions.to_csv(ARTIFACT_DIR / "player_prediction_details_fast.csv", index=False)

    # print the final results table to the console
    print("\nFINAL RESULTS")
    print(results_df)

    # confirm where the output files were saved
    print("\nSaved:")
    print(f"- {ARTIFACT_DIR / 'player_results_fast.csv'}")
    print(f"- {ARTIFACT_DIR / 'player_prediction_details_fast.csv'}")


# only run main() when this script is executed directly
# if this file is imported by another script, main() won't run automatically
if __name__ == "__main__":
    main()
