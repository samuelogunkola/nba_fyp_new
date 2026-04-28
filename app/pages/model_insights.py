# Path lets us point to files on disk in a way that works across operating systems
from pathlib import Path

# pandas is used to load and work with CSV data (rows and columns, like a spreadsheet)
import pandas as pd

# streamlit turns this Python script into an interactive web page
import streamlit as st

# shared UI helpers used on every page for consistent styling
from ui import apply_global_styles, render_sidebar, page_header, metric_card, glass_card, divider


# set the browser tab title, emoji icon, and stretch content to fill the full screen width
st.set_page_config(
    page_title="Model Insights",
    page_icon="📉",
    layout="wide",
)

# apply the shared CSS styles so this page looks consistent with the rest of the app
apply_global_styles()

# draw the left-hand sidebar navigation that appears on every page
render_sidebar()


# file paths pointing to the CSV files produced by each of the three ML experiments
# these CSVs contain the evaluation metrics (accuracy, MAE, R², etc.) for every model tested
WIN_RESULTS = Path("models/experiments/win/artifacts/win_results.csv")
SCORE_RESULTS = Path("models/experiments/score/artifacts/score_results.csv")
PLAYER_RESULTS = Path("models/experiments/player/artifacts/player_results_fast.csv")


# @st.cache_data tells streamlit to remember the result so it doesn't re-read
# the file from disk every single time the page refreshes
@st.cache_data
def load_csv(path: Path):
    # only read the file if it actually exists on disk, otherwise return None
    if path.exists():
        return pd.read_csv(path)
    return None


# load all three results CSVs up front — any that are missing will be None
win_df = load_csv(WIN_RESULTS)
score_df = load_csv(SCORE_RESULTS)
player_df = load_csv(PLAYER_RESULTS)


# render the top banner with the page title, description, and tag pills
page_header(
    "📉",
    "Model Insights",
    "Compare trained model performance across win prediction, score prediction, and player stat prediction.",
    ["Model Evaluation", "Experiment Results", "Performance Analysis"],
)

# remind the reader that these numbers came from the full local experiments,
# not the lightweight demo datasets used in the deployed version
st.info(
    "These metrics come from the full local machine learning experiments. "
    "The hosted app uses lightweight demo datasets to keep the usability-test version fast and accessible."
)


# top-level summary cards — one headline number from each of the three model families
st.subheader("Overall Performance Summary")

# four equal columns for four summary cards
c1, c2, c3, c4 = st.columns(4)

# card 1: the single best-performing win prediction model by F1 score
with c1:
    if win_df is not None:
        # sort all win models by F1 score descending and grab the top row
        best_f1 = win_df.sort_values("f1", ascending=False).iloc[0]
        metric_card("Best Win Model", best_f1["model_name"], f"F1: {best_f1['f1']:.3f}")
    else:
        # CSV wasn't found — show a prompt instead of crashing
        metric_card("Best Win Model", "Missing", "Run win experiment")

# card 2: the Ridge regression result for predicting home team points
with c2:
    if score_df is not None:
        # filter to just Ridge model results for the home_pts target column
        ridge_home = score_df[(score_df["model_name"] == "Ridge") & (score_df["target"] == "home_pts")]
        if not ridge_home.empty:
            metric_card("Best Score Model", "Ridge", f"Home MAE: {ridge_home.iloc[0]['mae']:.2f}")
        else:
            metric_card("Best Score Model", "N/A", "No Ridge result")
    else:
        metric_card("Best Score Model", "Missing", "Run score experiment")

# card 3: the R² score for the player points prediction model
with c3:
    if player_df is not None:
        # grab the first row where the prediction target is points
        pts = player_df[player_df["target"] == "pts"].iloc[0]
        metric_card("Best Player Model", "Ridge", f"PTS R²: {pts['r2']:.3f}")
    else:
        metric_card("Best Player Model", "Missing", "Run player experiment")

# card 4: reminder of how evaluation was done (time-based split, not random)
with c4:
    metric_card("Evaluation Type", "Time Split", "Pregame leakage-safe testing")


# horizontal rule to separate the summary row from the detailed sections below
divider()


# detailed results for the win prediction experiment
st.subheader("🏆 Win Prediction Models")

if win_df is None:
    # let the user know exactly which file is missing so they can fix it
    st.warning(f"Missing file: `{WIN_RESULTS}`")
else:
    # find the best model for each of the three key classification metrics independently
    # a different model might win on each metric, which is worth showing
    best_accuracy = win_df.sort_values("accuracy", ascending=False).iloc[0]
    best_f1 = win_df.sort_values("f1", ascending=False).iloc[0]
    best_auc = win_df.sort_values("roc_auc", ascending=False).iloc[0]

    # three cards showing which model topped each metric
    w1, w2, w3 = st.columns(3)

    with w1:
        # accuracy = percentage of games the model predicted correctly
        metric_card("Best Accuracy", best_accuracy["model_name"], f"{best_accuracy['accuracy']:.3f}")

    with w2:
        # F1 = harmonic mean of precision and recall — best single metric when classes are balanced
        metric_card("Best F1", best_f1["model_name"], f"{best_f1['f1']:.3f}")

    with w3:
        # ROC-AUC measures how well the model separates wins from losses across all thresholds
        metric_card("Best ROC-AUC", best_auc["model_name"], f"{best_auc['roc_auc']:.3f}")

    # grouped bar chart comparing all win models across all five metrics at once
    st.markdown("#### Win Model Comparison")

    # pivot the dataframe so model names are on the x-axis and metrics are the bar groups
    chart_df = win_df.set_index("model_name")[["accuracy", "precision", "recall", "f1", "roc_auc"]]
    st.bar_chart(chart_df)

    # also show the full raw numbers in a scrollable table for anyone who wants the detail
    st.dataframe(win_df, use_container_width=True)

    # plain-English takeaway card summarising what the numbers mean
    glass_card(
        "Interpretation",
        "Gradient Boosting achieved the strongest F1 score, making it the best model when balancing precision and recall. Random Forest achieved similar accuracy, while Logistic Regression provided a useful linear baseline.",
    )


divider()


# detailed results for the score prediction experiment
st.subheader("📈 Score Prediction Models")

if score_df is None:
    st.warning(f"Missing file: `{SCORE_RESULTS}`")
else:
    # focus on Ridge regression results since it was the best-performing score model
    ridge_df = score_df[score_df["model_name"] == "Ridge"].copy()

    if not ridge_df.empty:
        # four cards, one for each regression target (home pts, away pts, spread, total)
        s1, s2, s3, s4 = st.columns(4)

        # small helper to safely pull a single metric value for a given target column
        # returns None if that combination doesn't exist in the results CSV
        def get_ridge_metric(target, metric):
            row = ridge_df[ridge_df["target"] == target]
            if row.empty:
                return None
            return row.iloc[0][metric]

        # look up the MAE (mean absolute error) for each of the four targets
        # MAE tells us how many points off the prediction was on average
        home_mae = get_ridge_metric("home_pts", "mae")
        away_mae = get_ridge_metric("away_pts", "mae")
        spread_mae = get_ridge_metric("point_spread", "mae")
        total_mae = get_ridge_metric("total_points", "mae")

        with s1:
            # format to 2 decimal places if the value exists, otherwise show N/A
            metric_card("Home MAE", f"{home_mae:.2f}" if home_mae is not None else "N/A", "Ridge")

        with s2:
            metric_card("Away MAE", f"{away_mae:.2f}" if away_mae is not None else "N/A", "Ridge")

        with s3:
            metric_card("Spread MAE", f"{spread_mae:.2f}" if spread_mae is not None else "N/A", "Ridge")

        with s4:
            metric_card("Total MAE", f"{total_mae:.2f}" if total_mae is not None else "N/A", "Ridge")

    # grouped bar chart: targets on the x-axis, each model's MAE as a separate bar
    st.markdown("#### Score Model MAE Comparison")

    # pivot_table reshapes the dataframe so we can compare models side by side per target
    mae_chart = score_df.pivot_table(
        index="target",
        columns="model_name",
        values="mae",
        aggfunc="mean",
    )

    st.bar_chart(mae_chart)

    # full raw results table for transparency
    st.dataframe(score_df, use_container_width=True)

    # interpretation card explaining why the scores look the way they do
    glass_card(
        "Interpretation",
        "Score prediction was the most difficult regression task. Ridge Regression performed best overall, suggesting that the rolling pregame features contain mostly linear signal. The low R² values are realistic because exact NBA scores are highly variable.",
    )


divider()


# detailed results for the player stat prediction experiment
st.subheader("👤 Player Stat Prediction Models")

if player_df is None:
    st.warning(f"Missing file: `{PLAYER_RESULTS}`")
else:
    # three cards, one for each stat being predicted (points, rebounds, assists)
    p1, p2, p3 = st.columns(3)

    # pull out the result row for each of the three prediction targets
    pts = player_df[player_df["target"] == "pts"].iloc[0]
    reb = player_df[player_df["target"] == "reb"].iloc[0]
    ast = player_df[player_df["target"] == "ast"].iloc[0]

    with p1:
        # MAE = average prediction error in points; R² = how much variance the model explains
        metric_card("Points Model", f"MAE {pts['mae']:.2f}", f"R²: {pts['r2']:.3f}")

    with p2:
        metric_card("Rebounds Model", f"MAE {reb['mae']:.2f}", f"R²: {reb['r2']:.3f}")

    with p3:
        metric_card("Assists Model", f"MAE {ast['mae']:.2f}", f"R²: {ast['r2']:.3f}")

    # bar chart comparing R² scores across all three targets
    # R² closer to 1.0 means the model explains more of the variation in the stat
    st.markdown("#### Player Stat R² Comparison")

    r2_chart = player_df.set_index("target")[["r2"]]
    st.bar_chart(r2_chart)

    # separate bar chart for MAE so the scale doesn't get squashed against the R² values
    st.markdown("#### Player Stat MAE Comparison")

    mae_chart = player_df.set_index("target")[["mae"]]
    st.bar_chart(mae_chart)

    # full raw numbers for anyone who wants to dig in
    st.dataframe(player_df, use_container_width=True)

    glass_card(
        "Interpretation",
        "Player stat prediction produced the strongest regression results. Points achieved the highest R², while rebounds and assists achieved very low average errors. This suggests recent player-level rolling features provide strong predictive signal.",
    )


divider()


# final section tying together what all the results mean from an academic perspective
st.subheader("Academic Evaluation Summary")

# two cards giving a broader interpretation of the overall findings
a1, a2 = st.columns(2)

with a1:
    # explains why some tasks were harder than others
    glass_card(
        "Model Difficulty",
        "The results show that prediction difficulty varies by task. Exact score prediction is hardest because basketball scoring has high variance, while player stat prediction benefits from stable individual historical patterns.",
    )

with a2:
    # explains why certain algorithm choices made sense for each task
    glass_card(
        "Model Selection",
        "Ridge Regression was effective for regression tasks because rolling averages and efficiency metrics provide mainly linear signal. Gradient Boosting performed best for win classification due to its ability to capture non-linear matchup patterns.",
    )
