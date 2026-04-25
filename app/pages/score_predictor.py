from pathlib import Path

import pandas as pd
import streamlit as st

from ui import apply_global_styles, render_sidebar, page_header, metric_card, glass_card, divider


st.set_page_config(page_title="Score Predictor", page_icon="📈", layout="wide")

apply_global_styles()
render_sidebar()


SCORE_RESULTS_PATH = Path("models/experiments/score/artifacts/score_results.csv")
SCORE_DETAILS_PATH = Path("models/experiments/score/artifacts/score_prediction_details.csv")


@st.cache_data
def load_csv(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None


results_df = load_csv(SCORE_RESULTS_PATH)
details_df = load_csv(SCORE_DETAILS_PATH)


page_header(
    "📈",
    "Score Predictor",
    "Explore saved NBA score predictions and compare projected scores against actual outcomes.",
    ["Regression", "Ridge Model", "Score Forecasting", "Pregame Features"],
)


if details_df is None:
    st.error(f"Missing file: `{SCORE_DETAILS_PATH}`")
    st.stop()


# ============================================================
# FIX TEAM COLUMN DUPLICATES
# ============================================================

if "home_team" not in details_df.columns:
    if "home_team_x" in details_df.columns:
        details_df["home_team"] = details_df["home_team_x"]
    elif "home_team_y" in details_df.columns:
        details_df["home_team"] = details_df["home_team_y"]

if "away_team" not in details_df.columns:
    if "away_team_x" in details_df.columns:
        details_df["away_team"] = details_df["away_team_x"]
    elif "away_team_y" in details_df.columns:
        details_df["away_team"] = details_df["away_team_y"]


has_team_names = {"home_team", "away_team"}.issubset(details_df.columns)


# ============================================================
# DATE PARSING
# ============================================================

if "date" in details_df.columns:
    details_df["date"] = pd.to_datetime(details_df["date"], errors="coerce")


# ============================================================
# MODEL SUMMARY
# ============================================================

st.subheader("Score Model Summary")

if results_df is not None:
    ridge = results_df[results_df["model_name"] == "Ridge"]

    def get_val(target):
        row = ridge[ridge["target"] == target]
        return row.iloc[0]["mae"] if not row.empty else None

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Home MAE", f"{get_val('home_pts'):.2f}" if get_val("home_pts") else "N/A", "")

    with c2:
        metric_card("Away MAE", f"{get_val('away_pts'):.2f}" if get_val("away_pts") else "N/A", "")

    with c3:
        metric_card("Spread MAE", f"{get_val('point_spread'):.2f}" if get_val("point_spread") else "N/A", "")

    with c4:
        metric_card("Total MAE", f"{get_val('total_points'):.2f}" if get_val("total_points") else "N/A", "")


divider()


# ============================================================
# FILTERS
# ============================================================

st.subheader("Game Explorer")

filtered_df = details_df.copy()

if has_team_names:
    teams = sorted(
        set(details_df["home_team"].dropna().unique()) |
        set(details_df["away_team"].dropna().unique())
    )

    f1, f2, f3 = st.columns(3)

    with f1:
        home_filter = st.selectbox("Home Team", ["Any"] + teams)

    with f2:
        away_filter = st.selectbox("Away Team", ["Any"] + teams)

    with f3:
        sort_option = st.selectbox(
            "Sort by",
            ["Most recent", "Lowest total error", "Highest total error"]
        )

    if home_filter != "Any":
        filtered_df = filtered_df[filtered_df["home_team"] == home_filter]

    if away_filter != "Any":
        filtered_df = filtered_df[filtered_df["away_team"] == away_filter]

else:
    st.warning("Team names not found — using Game ID selection only.")

    sort_option = st.selectbox(
        "Sort by",
        ["Most recent", "Lowest total error", "Highest total error"]
    )


# ============================================================
# SORTING
# ============================================================

if sort_option == "Most recent" and "date" in filtered_df.columns:
    filtered_df = filtered_df.sort_values("date", ascending=False)
elif sort_option == "Lowest total error":
    filtered_df = filtered_df.sort_values("total_error", ascending=True)
elif sort_option == "Highest total error":
    filtered_df = filtered_df.sort_values("total_error", ascending=False)


if filtered_df.empty:
    st.warning("No games found.")
    st.stop()


# ============================================================
# GAME SELECTOR
# ============================================================

def game_label(row):
    if has_team_names:
        label = f"{row['away_team']} @ {row['home_team']}"
    else:
        label = f"Game {row['gameid']}"

    if pd.notna(row.get("date")):
        label += f" | {pd.to_datetime(row['date']).date()}"

    return label


filtered_df["label"] = filtered_df.apply(game_label, axis=1)

selected = st.selectbox("Select Game", filtered_df["label"])

row = filtered_df[filtered_df["label"] == selected].iloc[0]


# ============================================================
# OUTPUT
# ============================================================

st.subheader("Prediction")

pred_home = row["pred_home_pts"]
pred_away = row["pred_away_pts"]
actual_home = row["home_pts"]
actual_away = row["away_pts"]

c1, c2 = st.columns(2)

with c1:
    metric_card("Home Score", f"{pred_home:.1f}", f"Actual: {actual_home}")

with c2:
    metric_card("Away Score", f"{pred_away:.1f}", f"Actual: {actual_away}")


divider()


# ============================================================
# ERROR
# ============================================================

st.subheader("Errors")

e1, e2 = st.columns(2)

with e1:
    metric_card("Home Error", f"{abs(pred_home - actual_home):.1f}", "")

with e2:
    metric_card("Away Error", f"{abs(pred_away - actual_away):.1f}", "")


divider()


# ============================================================
# INTERPRETATION
# ============================================================

st.subheader("Interpretation")

glass_card(
    "Model Insight",
    "Score prediction is inherently noisy due to pace, variance, and in-game factors. Results should be interpreted as estimates, not exact outcomes.",
)