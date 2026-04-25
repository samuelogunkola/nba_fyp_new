from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from ui import apply_global_styles, render_sidebar, page_header, metric_card, glass_card, divider


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Win Predictor",
    page_icon="📊",
    layout="wide",
)

apply_global_styles()
render_sidebar()


# ============================================================
# PATHS
# ============================================================

WIN_DATA_PATH = Path("data/demo/win_demo.csv") #changed to demo data due to file size restrictions
WIN_RESULTS_PATH = Path("models/experiments/win/artifacts/win_results.csv")
WIN_MODEL_PATH = Path("models/experiments/win/artifacts/gradient_boosting_home_win.pkl")


# ============================================================
# LOADERS
# ============================================================

@st.cache_data
def load_data(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_results(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_resource
def load_model(path: Path):
    if path.exists():
        return joblib.load(path)
    return None


df = load_data(WIN_DATA_PATH)
results_df = load_results(WIN_RESULTS_PATH)
model = load_model(WIN_MODEL_PATH)


# ============================================================
# HEADER
# ============================================================

page_header(
    "📊",
    "Win Predictor",
    "Estimate the probability that the selected home team wins using a leakage-safe pregame machine learning model.",
    ["Classification", "Gradient Boosting", "F1 0.699", "Pregame Features"],
)


# ============================================================
# CHECKS
# ============================================================

if df is None:
    st.error(f"Missing dataset: `{WIN_DATA_PATH}`")
    st.stop()

if model is None:
    st.error(f"Missing trained model: `{WIN_MODEL_PATH}`")
    st.stop()

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

TARGET = "home_win"

DROP_COLS = ["gameid", "date", "season", TARGET]


def confidence_label(prob: float) -> str:
    if prob >= 0.70 or prob <= 0.30:
        return "High"
    if prob >= 0.60 or prob <= 0.40:
        return "Moderate"
    return "Low"


def prediction_label(prob: float) -> str:
    if prob >= 0.5:
        return "Home Win"
    return "Away Win"


def prediction_message(prob: float) -> str:
    if prob >= 0.70:
        return "Strong home-team favourite"
    if prob >= 0.60:
        return "Moderate home-team advantage"
    if prob >= 0.50:
        return "Slight home-team edge"
    if prob >= 0.40:
        return "Slight away-team edge"
    if prob >= 0.30:
        return "Moderate away-team advantage"
    return "Strong away-team favourite"


def is_leakage_col(col: str) -> bool:
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


def get_feature_columns(data: pd.DataFrame):
    features = []

    for col in data.columns:
        if col in DROP_COLS:
            continue

        if not pd.api.types.is_numeric_dtype(data[col]):
            continue

        if is_leakage_col(col):
            continue

        features.append(col)

    return features


def get_team_columns(data: pd.DataFrame):
    possible_home = ["home", "home_team", "home_name", "home_abbrev"]
    possible_away = ["away", "away_team", "away_name", "away_abbrev"]

    home_col = next((c for c in possible_home if c in data.columns), None)
    away_col = next((c for c in possible_away if c in data.columns), None)

    return home_col, away_col


home_col, away_col = get_team_columns(df)
feature_cols = get_feature_columns(df)


# ============================================================
# MODEL SUMMARY
# ============================================================

st.subheader("Model Overview")

m1, m2, m3, m4 = st.columns(4)

if results_df is not None and not results_df.empty:
    gb_row = results_df[results_df["model_name"] == "Gradient Boosting"]

    if not gb_row.empty:
        gb_row = gb_row.iloc[0]

        with m1:
            metric_card("Model", "Gradient Boosting", "Best F1 score")

        with m2:
            metric_card("Accuracy", f"{gb_row['accuracy']:.3f}", "Overall correct predictions")

        with m3:
            metric_card("F1 Score", f"{gb_row['f1']:.3f}", "Balance of precision and recall")

        with m4:
            metric_card("ROC-AUC", f"{gb_row['roc_auc']:.3f}", "Ranking quality")
    else:
        with m1:
            metric_card("Model", "Gradient Boosting", "Loaded model")
else:
    with m1:
        metric_card("Model", "Gradient Boosting", "Loaded model")

divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("Select Matchup")

if home_col is None or away_col is None:
    st.warning(
        "Team name columns were not found in the dataset. "
        "The page will use saved game IDs instead."
    )

    selected_game = st.selectbox(
        "Select Game ID",
        df["gameid"].dropna().astype(str).unique().tolist(),
    )

    selected_row = df[df["gameid"].astype(str) == selected_game].iloc[-1]

else:
    teams = sorted(set(df[home_col].dropna().unique()).union(set(df[away_col].dropna().unique())))

    input_col1, input_col2 = st.columns(2)

    with input_col1:
        home_team = st.selectbox("Home Team", teams, index=0)

    with input_col2:
        away_team = st.selectbox("Away Team", teams, index=1 if len(teams) > 1 else 0)

    if home_team == away_team:
        st.warning("Select two different teams.")
        st.stop()

    matchup_rows = df[
        (df[home_col] == home_team)
        & (df[away_col] == away_team)
    ].copy()

    if matchup_rows.empty:
        st.warning(
            "No exact historical matchup row found for this home/away combination. "
            "Try another pair or use an existing matchup from the dataset."
        )

        fallback = df[
            (df[home_col] == home_team)
            | (df[away_col] == away_team)
        ].copy()

        if fallback.empty:
            st.stop()

        selected_row = fallback.iloc[-1]
    else:
        selected_row = matchup_rows.iloc[-1]


# ============================================================
# PREDICTION
# ============================================================

X = pd.DataFrame([selected_row[feature_cols]])
X = X.replace([np.inf, -np.inf], np.nan)

prob = float(model.predict_proba(X)[0][1])
pred = int(prob >= 0.5)

outcome = prediction_label(prob)
confidence = confidence_label(prob)
message = prediction_message(prob)


# ============================================================
# OUTPUT CARDS
# ============================================================

st.subheader("Prediction Result")

r1, r2, r3, r4 = st.columns(4)

with r1:
    metric_card("Predicted Outcome", outcome, message)

with r2:
    metric_card("Home Win Probability", f"{prob:.1%}", f"{confidence} confidence")

with r3:
    metric_card("Away Win Probability", f"{1 - prob:.1%}", "Opposing probability")

with r4:
    metric_card("Model Used", "Gradient Boosting", "Best F1 model")


if prob >= 0.60:
    st.success(f"🏆 {message}: the model favours the home team with {prob:.1%} probability.")
elif prob <= 0.40:
    st.warning(f"⚠️ {message}: the model favours the away team with {(1 - prob):.1%} probability.")
else:
    st.info(f"⚖️ Close matchup: the model sees this as relatively balanced.")


divider()


# ============================================================
# PROBABILITY VISUAL
# ============================================================

st.subheader("Probability Breakdown")

prob_df = pd.DataFrame(
    {
        "Outcome": ["Home Win", "Away Win"],
        "Probability": [prob, 1 - prob],
    }
)

st.bar_chart(prob_df.set_index("Outcome"))


divider()


# ============================================================
# GAME CONTEXT
# ============================================================

st.subheader("Selected Matchup Context")

context_cols = st.columns(4)

with context_cols[0]:
    value = selected_row.get(home_col, "N/A") if home_col else "N/A"
    metric_card("Home Team", str(value), "Selected home side")

with context_cols[1]:
    value = selected_row.get(away_col, "N/A") if away_col else "N/A"
    metric_card("Away Team", str(value), "Selected away side")

with context_cols[2]:
    value = selected_row.get("season", "N/A")
    metric_card("Season", str(value), "Dataset season")

with context_cols[3]:
    value = selected_row.get("date", "N/A")
    if pd.notna(value) and value != "N/A":
        value = str(pd.to_datetime(value).date())
    metric_card("Reference Date", str(value), "Most recent matching row")


# ============================================================
# MODEL INTERPRETATION
# ============================================================

divider()

st.subheader("How to Interpret This Prediction")

info1, info2, info3 = st.columns(3)

with info1:
    glass_card(
        "Probability",
        "The output represents the estimated probability of the home team winning based on historical pregame features.",
    )

with info2:
    glass_card(
        "Confidence",
        "Predictions close to 50% are uncertain. Predictions above 60% or below 40% indicate stronger model preference.",
    )

with info3:
    glass_card(
        "Leakage Prevention",
        "The model uses rolling historical statistics only. Same-game box score statistics are excluded from prediction features.",
    )


divider()


# ============================================================
# FEATURE SNAPSHOT
# ============================================================

st.subheader("Feature Snapshot")

important_possible_features = [
    "diff_plus_minus_roll_mean_10",
    "diff_plus_minus_roll_mean_5",
    "home_plus_minus_roll_mean_10",
    "away_plus_minus_roll_mean_10",
    "diff_ts_pct_proxy_roll_mean_10",
    "diff_efg_pct_roll_mean_10",
]

available_snapshot = [c for c in important_possible_features if c in selected_row.index]

if available_snapshot:
    snapshot_df = pd.DataFrame(
        {
            "Feature": available_snapshot,
            "Value": [selected_row[c] for c in available_snapshot],
        }
    )

    st.dataframe(snapshot_df, use_container_width=True)
else:
    st.caption("No predefined feature snapshot columns were found in this dataset.")


# ============================================================
# RAW ROW TOGGLE
# ============================================================

with st.expander("View raw selected matchup row"):
    st.dataframe(pd.DataFrame([selected_row]), use_container_width=True)