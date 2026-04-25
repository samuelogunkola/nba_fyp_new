from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


WIN_DEMO = Path("data/demo/win_demo.csv")
PLAYER_DEMO = Path("data/demo/player_demo.csv")

WIN_MODEL = Path("models/experiments/win/artifacts/gradient_boosting_home_win.pkl")
PTS_MODEL = Path("models/experiments/player/artifacts/ridge_pts.pkl")
REB_MODEL = Path("models/experiments/player/artifacts/ridge_reb.pkl")
AST_MODEL = Path("models/experiments/player/artifacts/ridge_ast.pkl")


def build_preprocessor(cols):
    return ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("var", VarianceThreshold()),
            ("scale", StandardScaler()),
        ]), cols)
    ])


def create_win_model():
    df = pd.read_csv(WIN_DEMO)

    target = "home_win"
    drop_cols = ["gameid", "date", "season", target]

    def is_leak(col):
        allowed = "_roll_" in col or "_exp_" in col or col.startswith("diff_")
        if allowed:
            return False
        leak_terms = [
            "home_win", "away_win", "winner", "result",
            "home_pts", "away_pts", "point_spread",
            "total_points", "plus_minus",
        ]
        return any(term in col for term in leak_terms)

    features = [
        c for c in df.columns
        if c not in drop_cols
        and pd.api.types.is_numeric_dtype(df[c])
        and not is_leak(c)
    ]

    X = df[features].replace([np.inf, -np.inf], np.nan)
    y = df[target].astype(int)

    model = Pipeline([
        ("preprocessor", build_preprocessor(features)),
        ("model", GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            random_state=42,
        )),
    ])

    model.fit(X, y)
    WIN_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, WIN_MODEL)
    print("Saved:", WIN_MODEL)


def create_player_models():
    df = pd.read_csv(PLAYER_DEMO)

    targets = ["pts", "reb", "ast"]

    drop_cols = [
        "gameid", "date", "playerid", "player",
        "team", "home", "away", "position",
        "pts", "reb", "ast",
    ]

    features = [
        c for c in df.columns
        if c not in drop_cols
        and pd.api.types.is_numeric_dtype(df[c])
        and ("_roll_" in c or c == "min")
    ]

    X = df[features].replace([np.inf, -np.inf], np.nan)

    for target, path in [
        ("pts", PTS_MODEL),
        ("reb", REB_MODEL),
        ("ast", AST_MODEL),
    ]:
        y = df[target]

        model = Pipeline([
            ("prep", build_preprocessor(features)),
            ("model", Ridge(alpha=10.0)),
        ])

        model.fit(X, y)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        print("Saved:", path)


if __name__ == "__main__":
    create_win_model()
    create_player_models()