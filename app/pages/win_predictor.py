# Path lets us point to files in a clean, OS-independent way
from pathlib import Path

# numpy gives us maths tools — we use it here for the sigmoid function and clipping
import numpy as np

# pandas is used to load and work with CSV data (rows and columns, like a spreadsheet)
import pandas as pd

# streamlit is the framework that turns this Python script into an interactive web page
import streamlit as st

# shared UI helpers that keep the look and feel consistent across every page
from ui import apply_global_styles, render_sidebar, page_header, metric_card, glass_card, divider


# configure the browser tab — sets the title shown in the tab and the emoji favicon
# layout="wide" stretches the content across the full screen width
st.set_page_config(page_title="Win Predictor", page_icon="📊", layout="wide")

# apply the shared CSS styles so this page looks the same as all the others
apply_global_styles()

# draw the left-hand sidebar navigation that appears on every page
render_sidebar()


# path to the lightweight demo dataset used for the hosted version of this page
# the full dataset is too large to deploy, so a smaller sample is used instead
WIN_DATA_PATH = Path("data/demo/win_demo.csv")

# path to the CSV containing model evaluation metrics from the full local experiments
WIN_RESULTS_PATH = Path("models/experiments/win/artifacts/win_results.csv")


# @st.cache_data means streamlit remembers the result so it doesn't re-read the file every refresh
@st.cache_data
def load_csv(path: Path):
    # only try to read the file if it actually exists, otherwise return None safely
    if path.exists():
        return pd.read_csv(path)
    return None


# load both files up front before any UI is drawn
df = load_csv(WIN_DATA_PATH)           # the demo matchup dataset
results_df = load_csv(WIN_RESULTS_PATH)  # the model evaluation results


# render the top banner with the page title, description, and tag pills
page_header(
    "📊",
    "Win Predictor",
    "Explore a hosted demo win-probability interface using lightweight pregame data. Full trained model results are reported in Model Insights.",
    ["Hosted Demo", "Pregame Features", "Win Probability", "Full Results in Model Insights"],
)

# info box reminding the user that this is a demo interface, not the full trained model
st.info(
    "Hosted version note: this page uses lightweight demo data for reliable online usability testing. "
    "The full trained Gradient Boosting model evaluation is shown in Model Insights."
)

# if the demo dataset failed to load, there's nothing to show — stop the page here
if df is None:
    st.error(f"Missing demo dataset: `{WIN_DATA_PATH}`")
    st.stop()


# parse the date column into proper datetime objects so we can sort games in order
# errors="coerce" turns any unreadable dates into NaT (not-a-time) instead of crashing
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# sort oldest game first so iloc[-1] always gives us the most recent row for any matchup
df = df.sort_values("date").reset_index(drop=True)


def confidence_label(prob):
    # turn a raw probability into a human-readable confidence level
    # probabilities far from 50/50 (very high or very low) indicate a clear favourite
    if prob >= 0.70 or prob <= 0.30:
        return "High"      # model is strongly favouring one side
    if prob >= 0.60 or prob <= 0.40:
        return "Moderate"  # model has a lean but it's not decisive
    return "Low"           # too close to call — could go either way


def outcome_label(prob):
    # anything above 50% means the model thinks the home team is more likely to win
    return "Home Win" if prob >= 0.5 else "Away Win"


def estimate_probability(row):
    # this function estimates a win probability from the rolling pregame features
    # it's used as a lightweight inference method that doesn't require loading a large model file

    # these are the feature columns we'll use to build up a raw score
    # each one captures a different angle on recent team performance
    candidate_cols = [
        "diff_win_roll_mean_5",           # difference in 5-game win rates between teams
        "diff_plus_minus_roll_mean_5",    # difference in 5-game plus/minus averages
        "diff_plus_minus_roll_mean_10",   # same but over the last 10 games (longer trend)
        "home_win_roll_mean_5",           # home team's win rate over their last 5 games
        "home_plus_minus_roll_mean_5",    # home team's average point margin over 5 games
        "away_plus_minus_roll_mean_5",    # away team's average point margin over 5 games
    ]

    # start with a neutral score of zero before adding up the feature contributions
    score = 0.0

    for col in candidate_cols:
        # only include the column if it exists in this row and has a real value (not NaN)
        if col in row.index and pd.notna(row[col]):
            # scale each feature's contribution down by 0.05 so they don't dominate the score
            score += float(row[col]) * 0.05

    # apply the sigmoid function to squash the raw score into a 0–1 probability
    # sigmoid maps any real number to a value between 0 and 1, which is perfect for probabilities
    prob = 1 / (1 + np.exp(-score))

    # if we have a direct home win rate available, blend it with the sigmoid output
    # averaging the two gives us a more grounded estimate tied to actual win history
    if "home_win_roll_mean_5" in row.index and pd.notna(row["home_win_roll_mean_5"]):
        prob = (prob + float(row["home_win_roll_mean_5"])) / 2

    # clip the final probability to stay between 5% and 95%
    # this prevents the model from ever being 100% certain, which is never realistic
    return float(np.clip(prob, 0.05, 0.95))


# summary cards at the top explaining the mode this page runs in and its best metric
st.subheader("Model Summary")

c1, c2, c3 = st.columns(3)

with c1:
    # clarify that this is a demo inference mode, not the full trained Gradient Boosting model
    metric_card("Hosted Mode", "Demo Inference", "Lightweight Streamlit deployment")

with c2:
    if results_df is not None:
        # find the model with the highest F1 score from the local experiments
        best = results_df.sort_values("f1", ascending=False).iloc[0]
        metric_card("Best Model", str(best["model_name"]), f"F1: {best['f1']:.3f}")
    else:
        metric_card("Best Model", "N/A", "Results unavailable")

with c3:
    # show how many matchup rows are in the demo dataset
    metric_card("Dataset", f"{len(df):,} rows", "Demo matchup records")


divider()


# let the user pick which two teams they want a win probability estimate for
st.subheader("Select Matchup")

# check whether the dataframe has team name columns — the column might be called "home" or similar
home_col = "home" if "home" in df.columns else None
away_col = "away" if "away" in df.columns else None

if home_col and away_col:
    # build a combined sorted list of every team that appears as either home or away
    teams = sorted(set(df[home_col].dropna()).union(set(df[away_col].dropna())))

    # two dropdowns side by side — one for home, one for away
    s1, s2 = st.columns(2)

    with s1:
        home_team = st.selectbox("Home Team", teams)

    with s2:
        # default the away team to the second option so the page doesn't start with identical teams
        away_team = st.selectbox("Away Team", teams, index=1 if len(teams) > 1 else 0)

    # a matchup against yourself doesn't make sense — warn the user and stop
    if home_team == away_team:
        st.warning("Please select two different teams.")
        st.stop()

    # look for rows where this exact home vs away combination appears in the dataset
    matchup_rows = df[(df[home_col] == home_team) & (df[away_col] == away_team)].copy()

    if matchup_rows.empty:
        # if these two teams have never met in the demo data, fall back to any recent row
        # involving the selected home team so we still have pregame features to work with
        st.warning("No exact historical matchup found. Showing the most recent row involving the selected home team.")
        matchup_rows = df[(df[home_col] == home_team) | (df[away_col] == away_team)].copy()

    # if we still couldn't find anything at all, stop here
    if matchup_rows.empty:
        st.error("No matching rows found.")
        st.stop()

    # use the most recent game row from the matched results as the feature input
    row = matchup_rows.sort_values("date").iloc[-1]

else:
    # team name columns weren't found — let the user pick a game by its raw ID instead
    selected_game = st.selectbox("Select Game ID", df["gameid"].astype(str).tolist())
    row = df[df["gameid"].astype(str) == selected_game].iloc[-1]


# run the probability estimate using the selected row's pregame features
prob = estimate_probability(row)

# convert the raw probability into readable labels
outcome = outcome_label(prob)
confidence = confidence_label(prob)


# show the four key prediction outputs: outcome, home probability, away probability, and mode
st.subheader("Prediction Result")

r1, r2, r3, r4 = st.columns(4)

with r1:
    # "Home Win" or "Away Win" depending on which side the probability favours
    metric_card("Predicted Outcome", outcome, "Estimated result")

with r2:
    # home win probability formatted as a percentage (e.g. 63.4%)
    metric_card("Home Win Probability", f"{prob:.1%}", f"{confidence} confidence")

with r3:
    # away win probability is simply the complement of the home probability
    metric_card("Away Win Probability", f"{1 - prob:.1%}", "Opposing probability")

with r4:
    metric_card("Mode", "Hosted Demo", "No large model file required")


# show a contextual banner depending on how decisive the probability estimate is
if prob >= 0.60:
    # the model clearly favours the home team
    st.success(f"🏆 The model favours the home team with {prob:.1%} estimated probability.")
elif prob <= 0.40:
    # the model clearly favours the away team
    st.warning(f"⚠️ The model favours the away team with {(1 - prob):.1%} estimated probability.")
else:
    # anywhere between 40% and 60% is essentially a coin flip
    st.info("⚖️ This appears to be a close matchup.")


divider()


# simple bar chart showing the home and away win probabilities side by side
st.subheader("Probability Breakdown")

# build a small two-row dataframe for the chart — streamlit's bar_chart needs it indexed by label
prob_df = pd.DataFrame(
    {
        "Outcome": ["Home Win", "Away Win"],
        "Probability": [prob, 1 - prob],
    }
)

st.bar_chart(prob_df.set_index("Outcome"))


divider()


# context cards showing details about the matchup row that was used as input
st.subheader("Matchup Context")

m1, m2, m3, m4 = st.columns(4)

with m1:
    # the home team from the actual dataset row (may differ slightly from the user's selection
    # if we fell back to a nearby row)
    metric_card("Home Team", str(row.get("home", "N/A")), "Selected home side")

with m2:
    metric_card("Away Team", str(row.get("away", "N/A")), "Selected away side")

with m3:
    # which NBA season the reference row is from
    metric_card("Season", str(row.get("season", "N/A")), "Dataset season")

with m4:
    # the date of the most recent matching game row
    value = row.get("date", "N/A")
    if pd.notna(value):
        # convert to a plain date string (no time component) for clean display
        value = str(pd.to_datetime(value).date())
    metric_card("Reference Date", str(value), "Most recent matching row")


divider()


# three cards explaining the context and limitations of what this page shows
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


# collapsible section showing the raw feature values from the row used for the prediction
# useful for debugging or understanding exactly what data the probability estimate is based on
with st.expander("View selected matchup row"):
    st.dataframe(pd.DataFrame([row]), use_container_width=True)
