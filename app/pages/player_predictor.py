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
    page_title="Player Predictor",
    page_icon="👤",
    layout="wide",
)

apply_global_styles()
render_sidebar()


# ============================================================
# PATHS
# ============================================================

PLAYER_DATA_PATH = Path("data/demo/player_demo.csv") #changed to demo data due to file size restrictions
PLAYER_RESULTS_PATH = Path("models/experiments/player/artifacts/player_results_fast.csv")

PTS_MODEL_PATH = Path("models/experiments/player/artifacts/ridge_pts.pkl")
REB_MODEL_PATH = Path("models/experiments/player/artifacts/ridge_reb.pkl")
AST_MODEL_PATH = Path("models/experiments/player/artifacts/ridge_ast.pkl")


# ============================================================
# LOADERS
# ============================================================

@st.cache_data
def load_player_data(path: Path):
    if path.exists():
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df.sort_values("date").reset_index(drop=True)
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


df = load_player_data(PLAYER_DATA_PATH)
results_df = load_results(PLAYER_RESULTS_PATH)

pts_model = load_model(PTS_MODEL_PATH)
reb_model = load_model(REB_MODEL_PATH)
ast_model = load_model(AST_MODEL_PATH)


# ============================================================
# HEADER
# ============================================================

page_header(
    "👤",
    "Player Stat Predictor",
    "Predict player points, rebounds, and assists using recent rolling player performance features.",
    ["Regression", "Ridge Model", "PTS R² 0.642", "Player-Level Prediction"],
)


# ============================================================
# FILE CHECKS
# ============================================================

if df is None:
    st.error(f"Missing player dataset: `{PLAYER_DATA_PATH}`")
    st.stop()

if pts_model is None or reb_model is None or ast_model is None:
    st.error("One or more trained player models are missing.")
    st.write("Expected:")
    st.write(f"- `{PTS_MODEL_PATH}`")
    st.write(f"- `{REB_MODEL_PATH}`")
    st.write(f"- `{AST_MODEL_PATH}`")
    st.stop()


# ============================================================
# HELPERS
# ============================================================

TARGETS = ["pts", "reb", "ast"]

DROP_COLS = [
    "gameid",
    "date",
    "playerid",
    "player",
    "team",
    "home",
    "away",
    "position",
    "pts",
    "reb",
    "ast",
]


def get_features(data: pd.DataFrame):
    features = []

    for col in data.columns:
        if col in DROP_COLS:
            continue

        if not pd.api.types.is_numeric_dtype(data[col]):
            continue

        if "_roll_" in col or col == "min":
            features.append(col)

    return features


def confidence_from_minutes(minutes):
    if pd.isna(minutes):
        return "Unknown"
    if minutes >= 28:
        return "High"
    if minutes >= 18:
        return "Moderate"
    return "Low"


def form_label(value):
    if pd.isna(value):
        return "Unavailable"
    if value >= 20:
        return "High scoring form"
    if value >= 12:
        return "Moderate scoring form"
    return "Low scoring form"


features = get_features(df)


# ============================================================
# MODEL PERFORMANCE SUMMARY
# ============================================================

st.subheader("Model Performance")

if results_df is not None:
    r1, r2, r3 = st.columns(3)

    pts_row = results_df[results_df["target"] == "pts"].iloc[0]
    reb_row = results_df[results_df["target"] == "reb"].iloc[0]
    ast_row = results_df[results_df["target"] == "ast"].iloc[0]

    with r1:
        metric_card("Points Model", f"MAE {pts_row['mae']:.2f}", f"R²: {pts_row['r2']:.3f}")

    with r2:
        metric_card("Rebounds Model", f"MAE {reb_row['mae']:.2f}", f"R²: {reb_row['r2']:.3f}")

    with r3:
        metric_card("Assists Model", f"MAE {ast_row['mae']:.2f}", f"R²: {ast_row['r2']:.3f}")
else:
    st.warning(f"Missing results file: `{PLAYER_RESULTS_PATH}`")

divider()


# ============================================================
# PLAYER SELECTION
# ============================================================

st.subheader("Select Player")

search_col, team_col = st.columns([1.4, 1])

with search_col:
    search_text = st.text_input("Search player name", placeholder="e.g. LeBron, Curry, Jokic")

with team_col:
    teams = ["All Teams"] + sorted(df["team"].dropna().unique().tolist())
    selected_team = st.selectbox("Filter by team", teams)


filtered_players = df.copy()

if selected_team != "All Teams":
    filtered_players = filtered_players[filtered_players["team"] == selected_team]

if search_text:
    filtered_players = filtered_players[
        filtered_players["player"].str.contains(search_text, case=False, na=False)
    ]

player_names = sorted(filtered_players["player"].dropna().unique().tolist())

if not player_names:
    st.warning("No players found. Try a different search or team filter.")
    st.stop()

selected_player = st.selectbox("Select player", player_names)

player_rows = df[df["player"] == selected_player].copy()

if selected_team != "All Teams":
    team_specific_rows = player_rows[player_rows["team"] == selected_team]
    if not team_specific_rows.empty:
        player_rows = team_specific_rows

player_rows = player_rows.sort_values("date")

latest_row = player_rows.iloc[-1]


# ============================================================
# PREDICTION
# ============================================================

X = pd.DataFrame([latest_row[features]])
X = X.replace([np.inf, -np.inf], np.nan)

pred_pts = float(pts_model.predict(X)[0])
pred_reb = float(reb_model.predict(X)[0])
pred_ast = float(ast_model.predict(X)[0])

pred_pts = max(0, pred_pts)
pred_reb = max(0, pred_reb)
pred_ast = max(0, pred_ast)


# ============================================================
# OUTPUT
# ============================================================

st.subheader("Prediction Result")

o1, o2, o3, o4 = st.columns(4)

with o1:
    metric_card("Predicted Points", f"{pred_pts:.1f}", "Projected PTS")

with o2:
    metric_card("Predicted Rebounds", f"{pred_reb:.1f}", "Projected REB")

with o3:
    metric_card("Predicted Assists", f"{pred_ast:.1f}", "Projected AST")

with o4:
    minutes_ref = latest_row.get("min_roll_mean_5", latest_row.get("min", np.nan))
    metric_card("Confidence", confidence_from_minutes(minutes_ref), "Based on recent minutes")


if pred_pts >= 20:
    st.success(f"🔥 {selected_player} is projected for a strong scoring performance.")
elif pred_pts >= 12:
    st.info(f"📊 {selected_player} is projected for a moderate scoring performance.")
else:
    st.warning(f"⚖️ {selected_player} is projected for a lower scoring output.")


divider()


# ============================================================
# PLAYER CONTEXT
# ============================================================

st.subheader("Player Context")

c1, c2, c3, c4 = st.columns(4)

with c1:
    metric_card("Player", selected_player, "Selected athlete")

with c2:
    metric_card("Team", str(latest_row.get("team", "N/A")), "Latest team in dataset")

with c3:
    metric_card("Position", str(latest_row.get("position", "N/A")), "Listed position")

with c4:
    latest_date = latest_row.get("date", "N/A")
    if pd.notna(latest_date) and latest_date != "N/A":
        latest_date = str(pd.to_datetime(latest_date).date())
    metric_card("Reference Date", str(latest_date), "Most recent player row")


divider()


# ============================================================
# RECENT FORM
# ============================================================

st.subheader("Recent Form Snapshot")

form_cols = st.columns(4)

pts_form = latest_row.get("pts_roll_mean_5_x", latest_row.get("pts_roll_mean_5", np.nan))
reb_form = latest_row.get("reb_roll_mean_5_x", latest_row.get("reb_roll_mean_5", np.nan))
ast_form = latest_row.get("ast_roll_mean_5_x", latest_row.get("ast_roll_mean_5", np.nan))
min_form = latest_row.get("min_roll_mean_5", np.nan)

with form_cols[0]:
    metric_card("5-Game PTS Avg", f"{pts_form:.1f}" if pd.notna(pts_form) else "N/A", form_label(pts_form))

with form_cols[1]:
    metric_card("5-Game REB Avg", f"{reb_form:.1f}" if pd.notna(reb_form) else "N/A", "Recent rebounding form")

with form_cols[2]:
    metric_card("5-Game AST Avg", f"{ast_form:.1f}" if pd.notna(ast_form) else "N/A", "Recent playmaking form")

with form_cols[3]:
    metric_card("5-Game MIN Avg", f"{min_form:.1f}" if pd.notna(min_form) else "N/A", "Recent role size")


chart_data = pd.DataFrame(
    {
        "Stat": ["Points", "Rebounds", "Assists", "Minutes"],
        "Recent Average": [
            pts_form if pd.notna(pts_form) else 0,
            reb_form if pd.notna(reb_form) else 0,
            ast_form if pd.notna(ast_form) else 0,
            min_form if pd.notna(min_form) else 0,
        ],
        "Prediction": [pred_pts, pred_reb, pred_ast, 0],
    }
)

st.bar_chart(chart_data.set_index("Stat"))


divider()


# ============================================================
# PLAYER GAME HISTORY
# ============================================================

st.subheader("Recent Player Game History")

history_cols = [
    c for c in [
        "date",
        "team",
        "home",
        "away",
        "min",
        "pts",
        "reb",
        "ast",
        "position",
    ]
    if c in player_rows.columns
]

st.dataframe(
    player_rows[history_cols].tail(15).sort_values("date", ascending=False),
    use_container_width=True,
)


divider()


# ============================================================
# INTERPRETATION
# ============================================================

st.subheader("How to Interpret This Prediction")

i1, i2, i3 = st.columns(3)

with i1:
    glass_card(
        "Role and Minutes",
        "Recent minutes are one of the strongest contextual indicators for player output. Players with stable playing time usually produce more reliable predictions.",
    )

with i2:
    glass_card(
        "Recent Form",
        "The model uses rolling historical features, meaning recent scoring, rebounding, assists, efficiency, and usage trends influence the prediction.",
    )

with i3:
    glass_card(
        "Model Scope",
        "Predictions are based on historical box-score patterns. They do not currently account for real-time injuries, starting lineup changes, or betting market information.",
    )


with st.expander("View raw latest player feature row"):
    st.dataframe(pd.DataFrame([latest_row]), use_container_width=True)