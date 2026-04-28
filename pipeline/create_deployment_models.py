# Path lets us build file paths in a clean, OS-independent way
from pathlib import Path

# joblib saves and loads trained model objects to/from disk as .pkl files
import joblib

# numpy is used to replace infinity values with NaN before training
import numpy as np

# pandas is used to load the demo CSV files
import pandas as pd

# ColumnTransformer applies a preprocessing pipeline to a specific set of columns
from sklearn.compose import ColumnTransformer

# GradientBoostingClassifier: builds decision trees one at a time, each fixing the last one's errors
# used here as the win prediction model because it performed best in the full experiments
from sklearn.ensemble import GradientBoostingClassifier

# VarianceThreshold removes features that barely change across rows — constant columns are useless
from sklearn.feature_selection import VarianceThreshold

# SimpleImputer fills missing values using the median of each column in the training data
from sklearn.impute import SimpleImputer

# Ridge: linear regression with L2 regularisation — used for the player stat prediction models
from sklearn.linear_model import Ridge

# Pipeline chains preprocessing and modelling steps into one object so they always run together
from sklearn.pipeline import Pipeline

# StandardScaler rescales features to zero mean and unit variance — needed for Ridge and Gradient Boosting
from sklearn.preprocessing import StandardScaler


# paths to the lightweight demo CSV files used for training the deployment models
# the full datasets are too large to deploy, so these smaller samples are used instead
WIN_DEMO = Path("data/demo/win_demo.csv")
PLAYER_DEMO = Path("data/demo/player_demo.csv")

# paths where the trained deployment models will be saved
# the dashboard loads these .pkl files at runtime to make predictions
WIN_MODEL = Path("models/experiments/win/artifacts/gradient_boosting_home_win.pkl")
PTS_MODEL = Path("models/experiments/player/artifacts/ridge_pts.pkl")
REB_MODEL = Path("models/experiments/player/artifacts/ridge_reb.pkl")
AST_MODEL = Path("models/experiments/player/artifacts/ridge_ast.pkl")


# builds a preprocessing pipeline that imputes missing values, removes constant features, and scales
# wrapping these steps in a ColumnTransformer means they're only applied to the specified columns
# and any other columns are discarded — this keeps inference clean and predictable
def build_preprocessor(cols):
    return ColumnTransformer([
        ("num", Pipeline([
            # step 1: fill any NaN values with the median of each column
            ("impute", SimpleImputer(strategy="median")),
            # step 2: drop any columns where every value is identical — they carry no signal
            ("var", VarianceThreshold()),
            # step 3: rescale to zero mean and unit variance
            ("scale", StandardScaler()),
        ]), cols)
    ])


# trains a Gradient Boosting classifier on the demo win dataset and saves it to disk
# this is the model the win predictor page loads when the app is running in the deployed environment
def create_win_model():
    # load the lightweight win demo dataset
    df = pd.read_csv(WIN_DEMO)

    # the column we're trying to predict: 1 = home team won, 0 = away team won
    target = "home_win"

    # columns to exclude from the feature matrix — identifiers and the target itself
    drop_cols = ["gameid", "date", "season", target]

    def is_leak(col):
        # returns True if this column looks like it would reveal the game's outcome
        # rolling and exponential moving average features are safe — they only look backwards
        # columns starting with "diff_" are pre-game matchup differences, also safe
        allowed = "_roll_" in col or "_exp_" in col or col.startswith("diff_")
        if allowed:
            return False

        # any column containing these strings would tell the model the result before it predicts
        # e.g. "home_pts" is the actual score — only known after the game ends
        leak_terms = [
            "home_win", "away_win", "winner", "result",
            "home_pts", "away_pts", "point_spread",
            "total_points", "plus_minus",
        ]
        return any(term in col for term in leak_terms)

    # build the feature list: numeric, not in the drop list, and not suspicious leakage
    features = [
        c for c in df.columns
        if c not in drop_cols
        and pd.api.types.is_numeric_dtype(df[c])
        and not is_leak(c)
    ]

    # extract the feature matrix and replace any infinity values with NaN
    X = df[features].replace([np.inf, -np.inf], np.nan)

    # cast the target to integer so the classifier sees clean 0s and 1s
    y = df[target].astype(int)

    # build a Pipeline that combines preprocessing and the Gradient Boosting classifier
    # the same settings as the best model found in the full local experiments
    model = Pipeline([
        ("preprocessor", build_preprocessor(features)),
        ("model", GradientBoostingClassifier(
            n_estimators=100,    # number of trees to build sequentially
            learning_rate=0.05, # how much each new tree corrects the previous ones
            max_depth=2,        # shallow trees to avoid overfitting on the small demo dataset
            random_state=42,    # fixed seed so results are reproducible
        )),
    ])

    # train the full pipeline on the demo data
    model.fit(X, y)

    # create the output directory if it doesn't exist, then save the trained model
    WIN_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, WIN_MODEL)
    print("Saved:", WIN_MODEL)


# trains three Ridge regression models on the demo player dataset — one for each stat
# (points, rebounds, assists) — and saves each one to disk for the player predictor page
def create_player_models():
    # load the lightweight player demo dataset
    df = pd.read_csv(PLAYER_DEMO)

    # the three stats we want to predict
    targets = ["pts", "reb", "ast"]

    # columns to exclude from the feature matrix
    # includes all three targets because including a target as a feature would be leakage
    drop_cols = [
        "gameid", "date", "playerid", "player",
        "team", "home", "away", "position",
        "pts",   # actual points — leakage
        "reb",   # actual rebounds — leakage
        "ast",   # actual assists — leakage
    ]

    # keep only rolling average features and the minutes column
    # rolling features capture recent player form without revealing same-game results
    # minutes played is a strong proxy for a player's current role and workload
    features = [
        c for c in df.columns
        if c not in drop_cols
        and pd.api.types.is_numeric_dtype(df[c])
        and ("_roll_" in c or c == "min")
    ]

    # extract and clean the feature matrix
    X = df[features].replace([np.inf, -np.inf], np.nan)

    # train a separate Ridge model for each of the three prediction targets
    for target, path in [
        ("pts", PTS_MODEL),
        ("reb", REB_MODEL),
        ("ast", AST_MODEL),
    ]:
        # extract the target column for this iteration
        y = df[target]

        # build a fresh pipeline for each target — each model is completely independent
        # Ridge alpha=10 means moderate regularisation to prevent overfitting on demo data
        model = Pipeline([
            ("prep", build_preprocessor(features)),
            ("model", Ridge(alpha=10.0)),
        ])

        # train the pipeline on the full demo dataset
        model.fit(X, y)

        # create the output directory if needed and save the trained model
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        print("Saved:", path)


# run both model creation functions when this script is executed directly
if __name__ == "__main__":
    create_win_model()
    create_player_models()
