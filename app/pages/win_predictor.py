from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ui import apply_global_styles, render_sidebar, page_header, metric_card, glass_card, divider


st.set_page_config(page_title="Win Predictor", page_icon="📊", layout="wide")

apply_global_styles()
render_sidebar()


WIN_DATA_PATH = Path("data/demo/win_demo.csv")
WIN_RESULTS_PATH = Path("models/experiments/win/artifacts/win_results.csv")


@st.cache_data
def load_csv(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None


df = load_csv(WIN_DATA_PATH)
results_df = load_csv(WIN_RESULTS_PATH)


page_header(
    "📊",
    "Win Predictor",
    "Explore a hosted demo win-probability interface using lightweight pregame data. Full trained model results are reported in Model Insights.",
    ["Hosted Demo", "Pregame Features", "Win Probability", "Full Results in Model Insights"],
)

st.info(
    "Hosted version note: this page uses lightweight demo data for reliable online usability testing. "
    "The full trained Gradient Boosting model evaluation is shown in Model Insights."
)

if df is None:
    st.error(f"Missing demo dataset: `{WIN_DATA_PATH}`")
    st.stop()


df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.sort_values("date").reset_index(drop=True)


def confidence_label(prob):
    if prob >= 0.70 or prob <= 0.30:
        return "High"
    if prob >= 0.60 or prob <= 0.40:
        return "Moderate"
    return "Low"


def outcome_label(prob):
    return "Home Win" if prob >= 0.5 else "Away Win"


def estimate_probability(row):
    candidate_cols = [
        "diff_win_roll_mean_5",
        "diff_plus_minus_roll_mean_5",
        "diff_plus_minus_roll_mean_10",
        "home_win_roll_mean_5",
        "home_plus_minus_roll_mean_5",
        "away_plus_minus_roll_mean_5",
    ]

    score = 0.0

    for col in candidate_cols:
        if col in row.index and pd.notna(row[col]):
            score += float(row[col]) * 0.05

    prob = 1 / (1 + np.exp(-score))

    if "home_win_roll_mean_5" in row.index and pd.notna(row["home_win_roll_mean_5"]):
        prob = (prob + float(row["home_win_roll_mean_5"])) / 2

    return float(np.clip(prob, 0.05, 0.95))


st.subheader("Model Summary")

c1, c2, c3 = st.columns(3)

with c1:
    metric_card("Hosted Mode", "Demo Inference", "Lightweight Streamlit deployment")

with c2:
    if results_df is not None:
        best = results_df.sort_values("f1", ascending=False).iloc[0]
        metric_card("Best Model", str(best["model_name"]), f"F1: {best['f1']:.3f}")
    else:
        metric_card("Best Model", "N/A", "Results unavailable")

with c3:
    metric_card("Dataset", f"{len(df):,} rows", "Demo matchup records")


divider()


st.subheader("Select Matchup")

home_col = "home" if "home" in df.columns else None
away_col = "away" if "away" in df.columns else None

if home_col and away_col:
    teams = sorted(set(df[home_col].dropna()).union(set(df[away_col].dropna())))

    s1, s2 = st.columns(2)

    with s1:
        home_team = st.selectbox("Home Team", teams)

    with s2:
        away_team = st.selectbox("Away Team", teams, index=1 if len(teams) > 1 else 0)

    if home_team == away_team:
        st.warning("Please select two different teams.")
        st.stop()

    matchup_rows = df[(df[home_col] == home_team) & (df[away_col] == away_team)].copy()

    if matchup_rows.empty:
        st.warning("No exact historical matchup found. Showing the most recent row involving the selected home team.")
        matchup_rows = df[(df[home_col] == home_team) | (df[away_col] == away_team)].copy()

    if matchup_rows.empty:
        st.error("No matching rows found.")
        st.stop()

    row = matchup_rows.sort_values("date").iloc[-1]

else:
    selected_game = st.selectbox("Select Game ID", df["gameid"].astype(str).tolist())
    row = df[df["gameid"].astype(str) == selected_game].iloc[-1]


prob = estimate_probability(row)
outcome = outcome_label(prob)
confidence = confidence_label(prob)


st.subheader("Prediction Result")

r1, r2, r3, r4 = st.columns(4)

with r1:
    metric_card("Predicted Outcome", outcome, "Estimated result")

with r2:
    metric_card("Home Win Probability", f"{prob:.1%}", f"{confidence} confidence")

with r3:
    metric_card("Away Win Probability", f"{1 - prob:.1%}", "Opposing probability")

with r4:
    metric_card("Mode", "Hosted Demo", "No large model file required")


if prob >= 0.60:
    st.success(f"🏆 The model favours the home team with {prob:.1%} estimated probability.")
elif prob <= 0.40:
    st.warning(f"⚠️ The model favours the away team with {(1 - prob):.1%} estimated probability.")
else:
    st.info("⚖️ This appears to be a close matchup.")


divider()


st.subheader("Probability Breakdown")

prob_df = pd.DataFrame(
    {
        "Outcome": ["Home Win", "Away Win"],
        "Probability": [prob, 1 - prob],
    }
)

st.bar_chart(prob_df.set_index("Outcome"))


divider()


st.subheader("Matchup Context")

m1, m2, m3, m4 = st.columns(4)

with m1:
    metric_card("Home Team", str(row.get("home", "N/A")), "Selected home side")

with m2:
    metric_card("Away Team", str(row.get("away", "N/A")), "Selected away side")

with m3:
    metric_card("Season", str(row.get("season", "N/A")), "Dataset season")

with m4:
    value = row.get("date", "N/A")
    if pd.notna(value):
        value = str(pd.to_datetime(value).date())
    metric_card("Reference Date", str(value), "Most recent matching row")


divider()


st.subheader("Interpretation")

i1, i2, i3 = st.columns(3)

with i1:
    glass_card(
        "Deployment Mode",
        "This hosted version uses lightweight demo data to keep the web app fast and accessible for usability testing.",
    )

with i2:
    glass_card(
        "Full Model Evaluation",
        "The full trained model performance is available in Model Insights, including F1, accuracy, precision, recall, and ROC-AUC.",
    )

with i3:
    glass_card(
        "Academic Validity",
        "The full machine learning pipeline was trained and evaluated locally, while this hosted version prioritises reliable user access.",
    )


with st.expander("View selected matchup row"):
    st.dataframe(pd.DataFrame([row]), use_container_width=True)