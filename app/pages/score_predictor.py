# Path lets us build file paths in a way that works across operating systems
from pathlib import Path

# pandas is used to load and work with the prediction output CSVs (rows and columns like a spreadsheet)
import pandas as pd

# streamlit is the framework that turns this Python script into an interactive web page
import streamlit as st

# shared UI helpers that keep the look and feel consistent across every page
from ui import apply_global_styles, render_sidebar, page_header, metric_card, glass_card, divider


# configure the browser tab title, emoji icon, and full-width layout
st.set_page_config(page_title="Score Predictor", page_icon="📈", layout="wide")

# apply the shared CSS so this page matches the rest of the app visually
apply_global_styles()

# draw the left-hand sidebar navigation
render_sidebar()


# path to the CSV containing overall model evaluation metrics (MAE, R², etc.)
SCORE_RESULTS_PATH = Path("models/experiments/score/artifacts/score_results.csv")

# path to the CSV containing the actual game-by-game predictions vs real outcomes
# this is the main data source for the interactive game explorer on this page
SCORE_DETAILS_PATH = Path("models/experiments/score/artifacts/score_prediction_details.csv")


# @st.cache_data means streamlit remembers the result and won't re-read the file on every refresh
@st.cache_data
def load_csv(path: Path):
    # only try to read the file if it actually exists, otherwise return None gracefully
    if path.exists():
        return pd.read_csv(path)
    return None


# load both CSVs up front so they're ready before any UI is drawn
results_df = load_csv(SCORE_RESULTS_PATH)   # overall model performance metrics
details_df = load_csv(SCORE_DETAILS_PATH)   # game-level prediction details


# render the top banner with the page title, description, and tag pills
page_header(
    "📈",
    "Score Predictor",
    "Explore saved NBA score predictions and compare projected scores against actual outcomes.",
    ["Regression", "Ridge Model", "Score Forecasting", "Pregame Features"],
)

# brief note explaining what this page shows and where the data comes from
st.info(
    "This page displays saved score prediction outputs from the trained local experiments. "
    "It is used to explore prediction quality and model errors in the hosted app."
)

# if the game-level details file is missing there's nothing to explore — stop early
if details_df is None:
    st.error(f"Missing file: `{SCORE_DETAILS_PATH}`")
    st.stop()


# when two dataframes are joined in pandas, duplicate column names get an _x or _y suffix
# (e.g. "home_team_x" and "home_team_y" instead of just "home_team")
# this block consolidates them back into a single clean column name
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

# track whether we managed to find team name columns — used later to decide how to label games
has_team_names = {"home_team", "away_team"}.issubset(details_df.columns)


# parse the date column into proper datetime objects so we can sort games chronologically
# errors="coerce" turns any unparseable dates into NaT (not-a-time) instead of crashing
if "date" in details_df.columns:
    details_df["date"] = pd.to_datetime(details_df["date"], errors="coerce")


# top-level summary cards showing the Ridge model's MAE for each prediction target
st.subheader("Score Model Summary")

if results_df is not None:
    # filter to just Ridge model rows — it was the best-performing score model
    ridge = results_df[results_df["model_name"] == "Ridge"]

    # small helper to safely fetch the MAE for a specific target (e.g. "home_pts")
    # returns None if that target row doesn't exist so we can show "N/A" instead of crashing
    def get_val(target):
        row = ridge[ridge["target"] == target]
        return row.iloc[0]["mae"] if not row.empty else None

    # four cards, one per regression target
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        # MAE for home points — how many points off the model was on average for home teams
        metric_card("Home MAE", f"{get_val('home_pts'):.2f}" if get_val("home_pts") else "N/A", "")

    with c2:
        # MAE for away points
        metric_card("Away MAE", f"{get_val('away_pts'):.2f}" if get_val("away_pts") else "N/A", "")

    with c3:
        # MAE for point spread (home score minus away score)
        metric_card("Spread MAE", f"{get_val('point_spread'):.2f}" if get_val("point_spread") else "N/A", "")

    with c4:
        # MAE for total points (home score plus away score, often called the "over/under")
        metric_card("Total MAE", f"{get_val('total_points'):.2f}" if get_val("total_points") else "N/A", "")


divider()


# interactive section where the user can search and filter individual game predictions
st.subheader("Game Explorer")

# start with all games and narrow down based on what the user selects
filtered_df = details_df.copy()

if has_team_names:
    # build a sorted list of every team that appears in either the home or away column
    teams = sorted(
        set(details_df["home_team"].dropna().unique()) |
        set(details_df["away_team"].dropna().unique())
    )

    # three controls: filter by home team, filter by away team, and choose sort order
    f1, f2, f3 = st.columns(3)

    with f1:
        # "Any" means don't filter by home team
        home_filter = st.selectbox("Home Team", ["Any"] + teams)

    with f2:
        # "Any" means don't filter by away team
        away_filter = st.selectbox("Away Team", ["Any"] + teams)

    with f3:
        # let the user pick whether to see the most recent games or the best/worst predictions
        sort_option = st.selectbox(
            "Sort by",
            ["Most recent", "Lowest total error", "Highest total error"]
        )

    # apply the home team filter if the user picked something other than "Any"
    if home_filter != "Any":
        filtered_df = filtered_df[filtered_df["home_team"] == home_filter]

    # apply the away team filter if the user picked something other than "Any"
    if away_filter != "Any":
        filtered_df = filtered_df[filtered_df["away_team"] == away_filter]

else:
    # team name columns weren't found — fall back to showing games by ID only
    st.warning("Team names not found — using Game ID selection only.")

    # still allow the user to sort even without team names
    sort_option = st.selectbox(
        "Sort by",
        ["Most recent", "Lowest total error", "Highest total error"]
    )


# apply the chosen sort order to the filtered dataframe
if sort_option == "Most recent" and "date" in filtered_df.columns:
    # sort descending so the newest game appears first
    filtered_df = filtered_df.sort_values("date", ascending=False)
elif sort_option == "Lowest total error":
    # smallest error first — these are the games the model nailed
    filtered_df = filtered_df.sort_values("total_error", ascending=True)
elif sort_option == "Highest total error":
    # largest error first — these are the games the model struggled on most
    filtered_df = filtered_df.sort_values("total_error", ascending=False)


# if the filters removed every game, tell the user and stop rather than showing an empty page
if filtered_df.empty:
    st.warning("No games found.")
    st.stop()


def game_label(row):
    # build a human-readable label for each game to display in the dropdown
    if has_team_names:
        # standard NBA format: away team @ home team
        label = f"{row['away_team']} @ {row['home_team']}"
    else:
        # fall back to game ID if team names aren't available
        label = f"Game {row['gameid']}"

    # append the date to make it easier to identify a specific game
    if pd.notna(row.get("date")):
        label += f" | {pd.to_datetime(row['date']).date()}"

    return label


# add the label as a new column so we can use it to look up the selected row later
filtered_df["label"] = filtered_df.apply(game_label, axis=1)

# dropdown listing every game that survived the filters, in the chosen sort order
selected = st.selectbox("Select Game", filtered_df["label"])

# retrieve the full data row for whichever game the user selected
row = filtered_df[filtered_df["label"] == selected].iloc[0]


# show the predicted scores alongside the real outcomes for the chosen game
st.subheader("Prediction")

# pull the four key values out of the selected row
pred_home = row["pred_home_pts"]    # what the model predicted for the home team
pred_away = row["pred_away_pts"]    # what the model predicted for the away team
actual_home = row["home_pts"]       # what the home team actually scored
actual_away = row["away_pts"]       # what the away team actually scored

# two cards side by side: home prediction vs actual, and away prediction vs actual
c1, c2 = st.columns(2)

with c1:
    # show the predicted score in the heading and the real score in the subtitle
    metric_card("Home Score", f"{pred_home:.1f}", f"Actual: {actual_home}")

with c2:
    metric_card("Away Score", f"{pred_away:.1f}", f"Actual: {actual_away}")


divider()


# show how far off each prediction was — absolute error means we ignore the direction (over/under)
st.subheader("Errors")

e1, e2 = st.columns(2)

with e1:
    # abs() turns a negative error into a positive one so we just see the magnitude
    metric_card("Home Error", f"{abs(pred_home - actual_home):.1f}", "")

with e2:
    metric_card("Away Error", f"{abs(pred_away - actual_away):.1f}", "")


divider()


# closing card putting the errors in context so the user knows how to read them
st.subheader("Interpretation")

glass_card(
    "Model Insight",
    "Score prediction is inherently noisy due to pace, variance, and in-game factors. Results should be interpreted as estimates, not exact outcomes.",
)
