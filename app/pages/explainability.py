# Path lets us build file paths in a way that works on any operating system
from pathlib import Path

# matplotlib is used to build the bar chart that shows feature importance
import matplotlib.pyplot as plt

# pandas is used to load and manipulate CSV data (like a spreadsheet in code)
import pandas as pd

# streamlit is the framework that turns this Python script into a web app
import streamlit as st

# these are shared UI helper functions defined in ui.py for consistent styling
from ui import apply_global_styles, render_sidebar, page_header, metric_card, glass_card, divider


# set the browser tab title, icon, and use the full width of the screen
st.set_page_config(page_title="Explainability", page_icon="🔍", layout="wide")

# apply the custom CSS styles shared across all pages
apply_global_styles()

# draw the left sidebar navigation that appears on every page
render_sidebar()


# path to the folder where all SHAP output files (CSVs and images) are stored
SHAP_DIR = Path("models/experiments/win/shap_outputs")

# path to the CSV that contains model evaluation results (accuracy, F1 score, etc.)
WIN_RESULTS_PATH = Path("models/experiments/win/artifacts/win_results.csv")


# @st.cache_data means streamlit will remember the result of this function
# so if the same file is requested again it won't re-read it from disk
@st.cache_data
def load_csv(path: Path):
    # only try to read the file if a path was given and that file actually exists
    if path and path.exists():
        return pd.read_csv(path)
    # if the file doesn't exist just return None so we can handle it gracefully later
    return None


def find_first_existing(patterns):
    # loop through each glob pattern (e.g. "*top20*.csv") and search the SHAP folder
    for pattern in patterns:
        # glob finds all files that match the pattern — like a wildcard search
        matches = list(SHAP_DIR.glob(pattern))
        if matches:
            # return the first match we find, we only need one
            return matches[0]
    # if nothing matched any of the patterns, return None
    return None


def safe_feature_chart(df, feature_col, value_col):
    # make a copy so we don't accidentally modify the original dataframe
    chart_df = df.copy()

    # make sure the feature names are strings so matplotlib can label the y-axis properly
    chart_df[feature_col] = chart_df[feature_col].astype(str)

    # convert the importance values to numbers, turning any bad values into NaN
    chart_df[value_col] = pd.to_numeric(chart_df[value_col], errors="coerce")

    # drop any rows where the importance value is missing, then sort smallest to largest
    # (sorted ascending so the longest bar ends up at the top of a horizontal chart)
    chart_df = chart_df.dropna(subset=[value_col]).sort_values(value_col)

    # create a matplotlib figure — figsize sets the width and height in inches
    fig, ax = plt.subplots(figsize=(10, 6))

    # draw a horizontal bar chart: features on the y-axis, SHAP values on the x-axis
    ax.barh(chart_df[feature_col], chart_df[value_col])

    # label the axes so the chart is easy to read
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("Feature")
    ax.set_title("Top SHAP Feature Importance")

    # hand the finished figure to streamlit to render it on the page
    st.pyplot(fig)


# load the model evaluation results CSV (contains F1 scores, accuracy, etc.)
win_results = load_csv(WIN_RESULTS_PATH)

# look for a SHAP importance CSV — we try the most specific name first,
# then fall back to more generic names in case the file was saved differently
shap_csv_path = find_first_existing([
    "*top20*.csv",
    "*summary*.csv",
    "*shap*.csv",
])

# look for the SHAP summary plot image (shows each feature's impact distribution)
summary_img_path = find_first_existing([
    "*summary*.png",
])

# look for the global importance bar chart image
importance_img_path = find_first_existing([
    "*bar*.png",
    "*importance*.png",
])

# look for a waterfall plot image (explains one individual prediction)
waterfall_img_path = find_first_existing([
    "*waterfall*.png",
    "*example*.png",
])

# actually load the SHAP CSV into a dataframe so we can work with it
# shap_csv_path might be None if no file was found, load_csv handles that safely
shap_df = load_csv(shap_csv_path)


# render the top banner with the page title, description, and tags
page_header(
    "🔍",
    "Explainability",
    "Explore saved SHAP outputs from the win prediction model to understand which features influenced predictions.",
    ["SHAP", "Feature Importance", "Transparency", "Win Prediction"],
)

# show an info box explaining why outputs are precomputed rather than generated live
st.info(
    "Explainability outputs are precomputed and saved as CSV/image files. "
    "They are displayed here rather than regenerated live to keep the deployed app lightweight."
)


# section heading for the four summary metric cards at the top
st.subheader("Explainability Overview")

# split the page into 4 equal columns for the overview cards
c1, c2, c3, c4 = st.columns(4)

# card 1: what explainability method is being used
with c1:
    metric_card("Method", "SHAP", "Feature contribution analysis")

# card 2: which model is being explained
with c2:
    metric_card("Model Explained", "Win Predictor", "Tree-based classifier")

# card 3: show the Gradient Boosting model's F1 score from the results CSV
with c3:
    if win_results is not None:
        # filter the results dataframe to just the Gradient Boosting row
        gb = win_results[win_results["model_name"] == "Gradient Boosting"]
        if not gb.empty:
            # format the F1 score to 3 decimal places
            metric_card("Model F1", f"{gb.iloc[0]['f1']:.3f}", "Best F1 score")
        else:
            # Gradient Boosting row wasn't in the CSV
            metric_card("Model F1", "N/A", "Result unavailable")
    else:
        # the results CSV itself wasn't found
        metric_card("Model F1", "N/A", "Win results missing")

# card 4: confirm whether a SHAP CSV was found and show its filename
with c4:
    if shap_df is not None:
        metric_card("SHAP Output", "Found", shap_csv_path.name)
    else:
        metric_card("SHAP Output", "Missing", "Check shap_outputs folder")


# horizontal line to separate the overview cards from the content below
divider()


# if no SHAP CSV was found at all, show a helpful warning and stop the page early
if shap_df is None:
    st.warning(
        f"""
        No SHAP CSV was found in:

        `{SHAP_DIR}`

        Expected files such as:
        - `win_shap_top20.csv`
        - `win_shap_summary.csv`
        - `win_shap_bar.png`
        - `win_shap_waterfall_example.png`
        """
    )

    # show placeholder cards explaining what the page would display if data existed
    st.subheader("What SHAP Would Show")

    e1, e2, e3 = st.columns(3)

    with e1:
        glass_card(
            "Global Importance",
            "Ranks features by average influence across predictions, showing which pregame metrics the model relies on most.",
        )

    with e2:
        glass_card(
            "Local Explanations",
            "Shows which features pushed one specific prediction towards a home win or away win.",
        )

    with e3:
        glass_card(
            "Leakage Checking",
            "Helps confirm that the model is relying on meaningful rolling features rather than same-game results.",
        )

    # stop here — no point rendering the rest of the page without any data
    st.stop()


# work out which columns in the CSV hold feature names and importance values
# the column names vary depending on how the SHAP output was saved, so we check several options

# strip any accidental whitespace from column names first
shap_df.columns = [str(c).strip() for c in shap_df.columns]

# list of column names that might contain feature names, in order of preference
possible_feature_cols = [
    "feature",
    "Feature",
    "feature_name",
    "column",
    "Unnamed: 0",   # pandas sometimes uses this when the index was saved as a column
]

# list of column names that might contain the SHAP importance values, in order of preference
possible_value_cols = [
    "mean_abs_shap",
    "mean_abs_SHAP",
    "importance",
    "Importance",
    "mean_shap",
    "shap_value",
    "SHAP",
]

# pick the first matching feature column name that actually exists in the dataframe
feature_col = next((c for c in possible_feature_cols if c in shap_df.columns), None)

# pick the first matching value column name that actually exists in the dataframe
value_col = next((c for c in possible_value_cols if c in shap_df.columns), None)

# if none of the known feature column names matched, just use the very first column
if feature_col is None:
    feature_col = shap_df.columns[0]

# if none of the known value column names matched, fall back to the last numeric column
if value_col is None:
    numeric_cols = shap_df.select_dtypes(include="number").columns.tolist()
    value_col = numeric_cols[-1] if numeric_cols else None


# render the bar chart and table for global feature importance

st.subheader("Global Feature Importance")

# if we still couldn't find a numeric importance column, show a warning and dump the raw data
if value_col is None:
    st.warning("SHAP CSV found, but no numeric importance column could be detected.")
    st.dataframe(shap_df.head(50), use_column_width=True)
else:
    # make sure the importance column is actually numeric (coerce turns bad values into NaN)
    shap_df[value_col] = pd.to_numeric(shap_df[value_col], errors="coerce")

    # drop rows where importance is NaN so they don't break the chart
    shap_df = shap_df.dropna(subset=[value_col])

    # sort highest importance first so the most influential features appear at the top
    shap_df = shap_df.sort_values(value_col, ascending=False).reset_index(drop=True)

    # cap the slider max at 30 or however many features exist, whichever is smaller
    max_features = min(30, len(shap_df))

    # let the user choose how many features to show using a slider
    top_n = st.slider(
        "Number of features to display",
        min_value=5,
        max_value=max_features,
        value=min(15, max_features),   # default to 15 features (or fewer if the data is small)
    )

    # slice the dataframe to only keep the top N rows and the two relevant columns
    top_df = shap_df[[feature_col, value_col]].head(top_n)

    # draw the horizontal bar chart using our helper function
    safe_feature_chart(top_df, feature_col, value_col)

    # also show the raw numbers in a scrollable table below the chart
    st.dataframe(top_df, use_container_width=True)

    # pull out the single most important feature and its score for the summary cards
    top_feature = str(top_df.iloc[0][feature_col])
    top_value = float(top_df.iloc[0][value_col])

    # three summary cards below the chart
    t1, t2, t3 = st.columns(3)

    with t1:
        # name of the feature with the highest mean absolute SHAP value
        metric_card("Top Feature", top_feature, "Most influential feature")

    with t2:
        # its actual SHAP score, formatted to 4 decimal places
        metric_card("Top Importance", f"{top_value:.4f}", "Mean absolute SHAP value")

    with t3:
        # how many features the user currently has the slider set to show
        metric_card("Features Displayed", str(top_n), "Current view")


# horizontal divider before the image section
divider()


# display the saved SHAP plots if they were found earlier

st.subheader("SHAP Visualisations")

# put the two main plots side by side in two columns
img1, img2 = st.columns(2)

with img1:
    st.markdown("#### Summary Plot")
    # the summary plot shows how each feature's SHAP values are spread across all predictions
    if summary_img_path and summary_img_path.exists():
        st.image(str(summary_img_path), use_column_width=True)
    else:
        st.info("Summary plot image not found.")

with img2:
    st.markdown("#### Global Importance")
    # the importance plot ranks features by their average absolute SHAP value
    if importance_img_path and importance_img_path.exists():
        st.image(str(importance_img_path), use_column_width=True)
    else:
        st.info("Global importance image not found.")


# the waterfall plot is optional — only show this section if the image file exists
# it explains one individual prediction by showing which features pushed the score up or down
if waterfall_img_path and waterfall_img_path.exists():
    divider()
    st.subheader("Single Prediction Explanation")
    st.image(str(waterfall_img_path), use_column_width=True)


# horizontal divider before the final interpretation section
divider()


# plain-English summary cards to help the reader interpret the SHAP results

st.subheader("Interpretation")

# three cards giving a plain-language summary of what the SHAP outputs mean
i1, i2, i3 = st.columns(3)

with i1:
    glass_card(
        "Global Patterns",
        "The global SHAP results show which historical pregame features the model uses most frequently across predictions.",
    )

with i2:
    glass_card(
        "Local Reasoning",
        "The waterfall plot explains a single prediction by showing which features pushed the model towards or away from a home win.",
    )

with i3:
    glass_card(
        "Model Trust",
        "Explainability helps confirm that the model relies on basketball-relevant features rather than leaked same-game results.",
    )


# final green success banner at the bottom of the page
st.success(
    "SHAP explainability strengthens the system by making model reasoning transparent and easier to evaluate."
)
