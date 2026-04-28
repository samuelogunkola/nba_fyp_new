# Path lets us write file paths in a clean, OS-independent way
from pathlib import Path

# joblib is used to load the trained model files that were saved to disk as .pkl files
import joblib

# numpy gives us useful math tools, like handling infinity and NaN (not-a-number) values
import numpy as np

# pandas is used to load and work with the player dataset (rows and columns like a spreadsheet)
import pandas as pd

# streamlit is the framework that turns this script into an interactive web page
import streamlit as st

# shared UI helpers that keep styling consistent across every page in the app
from ui import apply_global_styles, render_sidebar, page_header, metric_card, glass_card, divider


# configure the browser tab — sets the title, favicon emoji, and full-width layout
st.set_page_config(
    page_title="Player Predictor",
    page_icon="👤",
    layout="wide",
)

# apply the shared CSS so this page looks the same as all the others
apply_global_styles()

# draw the left-hand sidebar navigation
render_sidebar()


# paths to all the data and model files this page needs
# using demo data here because the full player dataset is too large to deploy
PLAYER_DATA_PATH = Path("data/demo/player_demo.csv") #changed to demo data due to file size restrictions

# the CSV containing the evaluation metrics (MAE, R²) from the player model experiments
PLAYER_RESULTS_PATH = Path("models/experiments/player/artifacts/player_results_fast.csv")

# the three trained Ridge regression models — one for each stat we predict
PTS_MODEL_PATH = Path("models/experiments/player/artifacts/ridge_pts.pkl")
REB_MODEL_PATH = Path("models/experiments/player/artifacts/ridge_reb.pkl")
AST_MODEL_PATH = Path("models/experiments/player/artifacts/ridge_ast.pkl")


# @st.cache_data caches the result so streamlit doesn't re-read the CSV every refresh
@st.cache_data
def load_player_data(path: Path):
    if path.exists():
        df = pd.read_csv(path)
        # parse the date column so we can sort games chronologically
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        # sort oldest to newest so iloc[-1] always gives us the most recent game row
        return df.sort_values("date").reset_index(drop=True)
    return None


# separate loader for the results CSV — simpler since it doesn't need date parsing
@st.cache_data
def load_results(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None


# @st.cache_resource is used for models instead of @st.cache_data because models
# are objects (not plain data), and cache_resource keeps them in memory more efficiently
@st.cache_resource
def load_model(path: Path):
    if path.exists():
        # joblib.load deserialises the .pkl file back into a scikit-learn model object
        return joblib.load(path)
    return None


# load everything up front before rendering any UI
df = load_player_data(PLAYER_DATA_PATH)
results_df = load_results(PLAYER_RESULTS_PATH)

# load all three models — each one predicts a different stat
pts_model = load_model(PTS_MODEL_PATH)
reb_model = load_model(REB_MODEL_PATH)
ast_model = load_model(AST_MODEL_PATH)


# render the top banner with the page title, description, and tag pills
page_header(
    "👤",
    "Player Stat Predictor",
    "Predict player points, rebounds, and assists using recent rolling player performance features.",
    ["Regression", "Ridge Model", "PTS R² 0.642", "Player-Level Prediction"],
)

# brief intro message so the user knows what data the predictions are based on
st.info("Explore player stat estimates using lightweight demo data and recent rolling performance features.")


# check that the player dataset loaded correctly — nothing else on this page works without it
if df is None:
    st.error(f"Missing player dataset: `{PLAYER_DATA_PATH}`")
    st.stop()  # halt execution here so we don't get confusing downstream errors

# check that all three models loaded correctly
if pts_model is None or reb_model is None or ast_model is None:
    st.error("One or more trained player models are missing.")
    # tell the user exactly which files are expected so they can fix the problem
    st.write("Expected:")
    st.write(f"- `{PTS_MODEL_PATH}`")
    st.write(f"- `{REB_MODEL_PATH}`")
    st.write(f"- `{AST_MODEL_PATH}`")
    st.stop()


# the three columns we're trying to predict — used elsewhere as a reference list
TARGETS = ["pts", "reb", "ast"]

# columns that should never be used as model input features
# these are either identifiers, labels we're predicting, or post-game results
DROP_COLS = [
    "gameid",
    "date",
    "playerid",
    "player",
    "team",
    "home",
    "away",
    "position",
    "pts",   # actual points — would be data leakage to include this as a feature
    "reb",   # actual rebounds — same reason
    "ast",   # actual assists — same reason
]


def get_features(data: pd.DataFrame):
    # build the list of columns the model should actually use as inputs
    features = []

    for col in data.columns:
        # skip anything in our drop list (identifiers, targets, etc.)
        if col in DROP_COLS:
            continue

        # skip any column that isn't a number — models can't work with raw strings
        if not pd.api.types.is_numeric_dtype(data[col]):
            continue

        # only include rolling average columns and the minutes column
        # rolling features capture recent trends without leaking same-game results
        if "_roll_" in col or col == "min":
            features.append(col)

    return features


def confidence_from_minutes(minutes):
    # estimate how reliable the prediction is based on how many minutes the player typically plays
    # more minutes = more stable role = more predictable output
    if pd.isna(minutes):
        return "Unknown"
    if minutes >= 28:
        return "High"      # starter-level minutes
    if minutes >= 18:
        return "Moderate"  # rotation player
    return "Low"           # bench player or limited role


def form_label(value):
    # turn a rolling points average into a human-readable form description
    if pd.isna(value):
        return "Unavailable"
    if value >= 20:
        return "High scoring form"
    if value >= 12:
        return "Moderate scoring form"
    return "Low scoring form"


# build the feature list once using the full dataframe — every player row has the same columns
features = get_features(df)


# show how well the models performed during training so the user can judge the predictions
st.subheader("Model Performance")

if results_df is not None:
    # three cards, one per stat — MAE tells us average error, R² tells us explanatory power
    r1, r2, r3 = st.columns(3)

    # look up the result row for each prediction target
    pts_row = results_df[results_df["target"] == "pts"].iloc[0]
    reb_row = results_df[results_df["target"] == "reb"].iloc[0]
    ast_row = results_df[results_df["target"] == "ast"].iloc[0]

    with r1:
        # MAE = mean absolute error in points; R² = proportion of variance explained (0–1)
        metric_card("Points Model", f"MAE {pts_row['mae']:.2f}", f"R²: {pts_row['r2']:.3f}")

    with r2:
        metric_card("Rebounds Model", f"MAE {reb_row['mae']:.2f}", f"R²: {reb_row['r2']:.3f}")

    with r3:
        metric_card("Assists Model", f"MAE {ast_row['mae']:.2f}", f"R²: {ast_row['r2']:.3f}")
else:
    st.warning(f"Missing results file: `{PLAYER_RESULTS_PATH}`")

divider()


# let the user find the player they're interested in by name or team
st.subheader("Select Player")

# two search controls side by side — wider column for the name search
search_col, team_col = st.columns([1.4, 1])

with search_col:
    # free-text search so the user doesn't have to scroll through hundreds of names
    search_text = st.text_input("Search player name", placeholder="e.g. LeBron, Curry, Jokic")

with team_col:
    # pull the unique team names from the dataset and add an "All Teams" option at the top
    teams = ["All Teams"] + sorted(df["team"].dropna().unique().tolist())
    selected_team = st.selectbox("Filter by team", teams)


# start with a copy of the full dataset then narrow it down based on the filters
filtered_players = df.copy()

# if the user picked a specific team, keep only rows for that team
if selected_team != "All Teams":
    filtered_players = filtered_players[filtered_players["team"] == selected_team]

# if the user typed something in the search box, filter player names by that text
if search_text:
    filtered_players = filtered_players[
        filtered_players["player"].str.contains(search_text, case=False, na=False)
    ]

# get a sorted list of unique player names that survived the filters
player_names = sorted(filtered_players["player"].dropna().unique().tolist())

# if the filters removed everyone, show a message and stop rather than crashing
if not player_names:
    st.warning("No players found. Try a different search or team filter.")
    st.stop()

# dropdown so the user can pick the exact player they want from the filtered list
selected_player = st.selectbox("Select player", player_names)

# get all rows for the selected player from the full (unfiltered) dataset
player_rows = df[df["player"] == selected_player].copy()

# if the user also filtered by team, narrow down to that team's rows for this player
# (a player can appear on multiple teams across the dataset if they were traded)
if selected_team != "All Teams":
    team_specific_rows = player_rows[player_rows["team"] == selected_team]
    if not team_specific_rows.empty:
        player_rows = team_specific_rows

# make sure the rows are in chronological order so we can grab the most recent one
player_rows = player_rows.sort_values("date")

# the most recent game row is used as the input to the model
latest_row = player_rows.iloc[-1]


# build the model input by pulling the feature values from the player's most recent row
# wrap it in a DataFrame because scikit-learn's predict() expects a 2D structure
X = pd.DataFrame([latest_row[features]])

# replace any infinity values with NaN — these can occasionally appear in rolling calculations
# and would cause the model to produce a bad prediction
X = X.replace([np.inf, -np.inf], np.nan)

# run each model and extract the single predicted value as a plain Python float
pred_pts = float(pts_model.predict(X)[0])
pred_reb = float(reb_model.predict(X)[0])
pred_ast = float(ast_model.predict(X)[0])

# clamp predictions to zero — a player can't score negative points
pred_pts = max(0, pred_pts)
pred_reb = max(0, pred_reb)
pred_ast = max(0, pred_ast)


# display the three stat predictions and a confidence rating
st.subheader("Prediction Result")

# four columns: predicted PTS, REB, AST, and a confidence indicator
o1, o2, o3, o4 = st.columns(4)

with o1:
    metric_card("Predicted Points", f"{pred_pts:.1f}", "Projected PTS")

with o2:
    metric_card("Predicted Rebounds", f"{pred_reb:.1f}", "Projected REB")

with o3:
    metric_card("Predicted Assists", f"{pred_ast:.1f}", "Projected AST")

with o4:
    # try the 5-game rolling minutes average first; fall back to a raw minutes column if missing
    minutes_ref = latest_row.get("min_roll_mean_5", latest_row.get("min", np.nan))
    metric_card("Confidence", confidence_from_minutes(minutes_ref), "Based on recent minutes")


# show a contextual banner depending on how high the points prediction is
if pred_pts >= 20:
    st.success(f"🔥 {selected_player} is projected for a strong scoring performance.")
elif pred_pts >= 12:
    st.info(f"📊 {selected_player} is projected for a moderate scoring performance.")
else:
    st.warning(f"⚖️ {selected_player} is projected for a lower scoring output.")


divider()


# show some context about the player so the user knows who they're looking at
st.subheader("Player Context")

c1, c2, c3, c4 = st.columns(4)

with c1:
    metric_card("Player", selected_player, "Selected athlete")

with c2:
    # show the team from the most recent row in the dataset
    metric_card("Team", str(latest_row.get("team", "N/A")), "Latest team in dataset")

with c3:
    metric_card("Position", str(latest_row.get("position", "N/A")), "Listed position")

with c4:
    # show the date of the most recent game row used as the prediction input
    latest_date = latest_row.get("date", "N/A")
    if pd.notna(latest_date) and latest_date != "N/A":
        # convert to a plain date string (no time component) for cleaner display
        latest_date = str(pd.to_datetime(latest_date).date())
    metric_card("Reference Date", str(latest_date), "Most recent player row")


divider()


# show the player's recent rolling averages so the user can see what the model is working from
st.subheader("Recent Form Snapshot")

form_cols = st.columns(4)

# try the '_x' suffixed version first — pandas sometimes adds this suffix when merging datasets
# fall back to the plain column name if the suffixed version doesn't exist
pts_form = latest_row.get("pts_roll_mean_5_x", latest_row.get("pts_roll_mean_5", np.nan))
reb_form = latest_row.get("reb_roll_mean_5_x", latest_row.get("reb_roll_mean_5", np.nan))
ast_form = latest_row.get("ast_roll_mean_5_x", latest_row.get("ast_roll_mean_5", np.nan))
min_form = latest_row.get("min_roll_mean_5", np.nan)

with form_cols[0]:
    # show the 5-game rolling average for points, with a form label for context
    metric_card("5-Game PTS Avg", f"{pts_form:.1f}" if pd.notna(pts_form) else "N/A", form_label(pts_form))

with form_cols[1]:
    metric_card("5-Game REB Avg", f"{reb_form:.1f}" if pd.notna(reb_form) else "N/A", "Recent rebounding form")

with form_cols[2]:
    metric_card("5-Game AST Avg", f"{ast_form:.1f}" if pd.notna(ast_form) else "N/A", "Recent playmaking form")

with form_cols[3]:
    # average minutes gives a sense of how big the player's role is right now
    metric_card("5-Game MIN Avg", f"{min_form:.1f}" if pd.notna(min_form) else "N/A", "Recent role size")


# build a small dataframe so we can plot recent averages vs predictions side by side
chart_data = pd.DataFrame(
    {
        "Stat": ["Points", "Rebounds", "Assists", "Minutes"],
        "Recent Average": [
            pts_form if pd.notna(pts_form) else 0,
            reb_form if pd.notna(reb_form) else 0,
            ast_form if pd.notna(ast_form) else 0,
            min_form if pd.notna(min_form) else 0,
        ],
        # predictions for the same three stats; minutes has no prediction so we use 0
        "Prediction": [pred_pts, pred_reb, pred_ast, 0],
    }
)

# grouped bar chart — easy to see at a glance how the prediction compares to recent form
st.bar_chart(chart_data.set_index("Stat"))


divider()


# show the player's last 15 games so the user can see their recent box-score history
st.subheader("Recent Player Game History")

# only include columns that actually exist in the dataframe (avoids KeyError if a column is missing)
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

# take the 15 most recent games and display them newest first
st.dataframe(
    player_rows[history_cols].tail(15).sort_values("date", ascending=False),
    use_container_width=True,
)


divider()


# three cards explaining how to read and trust the predictions on this page
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


# collapsible section for debugging — lets a developer inspect the exact feature values
# that were fed into the model for the selected player
with st.expander("View raw latest player feature row"):
    st.dataframe(pd.DataFrame([latest_row]), use_container_width=True)
