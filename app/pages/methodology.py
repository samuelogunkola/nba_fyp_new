# streamlit is the framework that runs this script as an interactive web page
import streamlit as st

# shared UI helpers used across every page for consistent styling and layout
from ui import apply_global_styles, render_sidebar, page_header, metric_card, glass_card, divider


# configure the browser tab — sets the title shown in the tab and the emoji favicon
# layout="wide" makes the content stretch across the full screen instead of being centred
st.set_page_config(
    page_title="Methodology",
    page_icon="🧪",
    layout="wide",
)

# apply the shared CSS so this page looks the same as all the others
apply_global_styles()

# draw the left-hand sidebar that lets users navigate between pages
render_sidebar()


# render the top banner with the page title, subtitle, and tag pills
page_header(
    "🧪",
    "Methodology",
    "A structured overview of the full machine learning pipeline, including data preparation, feature engineering, modelling, leakage prevention, and evaluation.",
    ["System Design", "ML Pipeline", "Leakage-Safe", "Evaluation"],
)

# let the user know that the live app uses smaller demo data for speed,
# but the full pipeline was run locally with the complete datasets
st.info(
    "Deployment note: the hosted Streamlit version uses lightweight demo datasets for usability testing. "
    "The complete data processing, training, and evaluation pipeline was performed locally using the full datasets."
)


# high-level summary cards at the top so the reader gets the key facts at a glance
st.subheader("Methodology Summary")

# four equal columns, one card each
m1, m2, m3, m4 = st.columns(4)

# where the data came from
with m1:
    metric_card("Data Source", "NBA Historical Data", "Player & team box scores")

# how the train/test split was handled (time-based, not random, to avoid leakage)
with m2:
    metric_card("Validation", "Time-Based Split", "Past → Train, Future → Test")

# only historical stats known before tip-off were used as inputs
with m3:
    metric_card("Feature Type", "Pregame Only", "Rolling historical features")

# three separate prediction targets were built
with m4:
    metric_card("System Outputs", "3 Models", "Win, Score, Player Stats")


# horizontal rule to visually separate the summary from the pipeline steps
divider()


# walk through the six stages of the data pipeline in order
st.subheader("End-to-End Data Pipeline")

# first row of three cards covers collection, cleaning, and feature engineering
p1, p2, p3 = st.columns(3)

with p1:
    glass_card(
        "1. Data Collection",
        "Historical NBA datasets were sourced and combined into structured team-level and player-level datasets.",
    )

with p2:
    glass_card(
        "2. Data Cleaning",
        "Data was standardised by fixing date formats, resolving missing values, and ensuring consistent team and player records.",
    )

with p3:
    glass_card(
        "3. Feature Engineering",
        "Rolling statistical features were created to capture recent performance trends for teams and players.",
    )

# second row covers dataset construction, training, and deployment
p4, p5, p6 = st.columns(3)

with p4:
    glass_card(
        "4. Dataset Construction",
        "Separate datasets were built for win prediction, score prediction, and player stat prediction.",
    )

with p5:
    glass_card(
        "5. Model Training",
        "Multiple models were trained and evaluated using both classification and regression techniques.",
    )

with p6:
    glass_card(
        "6. Dashboard Deployment",
        "Model results, explainability outputs, and lightweight demo prediction interfaces were integrated into an interactive Streamlit application for usability testing.",
    )


divider()


# explain the feature engineering choices — rolling windows capture form without leaking future data
st.subheader("Feature Engineering Strategy")

f1, f2, f3 = st.columns(3)

# rolling averages smooth out single-game noise and represent recent team form
with f1:
    glass_card(
        "Rolling Averages",
        "3, 5, and 10-game rolling averages were used to represent recent performance trends.",
    )

# derived stats give the model richer signals than raw box-score numbers alone
with f2:
    glass_card(
        "Efficiency Metrics",
        "Derived metrics such as shooting percentages, efficiency ratings, and usage proxies were included.",
    )

# head-to-head difference features let the model compare the two teams directly
with f3:
    glass_card(
        "Matchup Features",
        "Differences between home and away teams were calculated to represent relative team strength.",
    )


# important note: every feature used must be available before the game tips off
st.info(
    "Only pregame features were used to ensure that predictions are realistic and usable before a game takes place."
)


divider()


# leakage is when the model accidentally sees information it shouldn't have at prediction time
# this section explains how that problem was discovered and fixed
st.subheader("Leakage Prevention")

l1, l2, l3 = st.columns(3)

# the first sign something was wrong: accuracy that looked too good to be true
with l1:
    glass_card(
        "Initial Issue",
        "Early models showed unrealistically high accuracy, indicating that future (post-game) information was leaking into the features.",
    )

# the fix: strip out any stat that's only known after the game ends
with l2:
    glass_card(
        "Leakage Removal",
        "All same-game statistics such as final scores, win labels, and plus-minus values were removed from the feature set.",
    )

# what the final clean pipeline looks like — every feature is shifted one game back
with l3:
    glass_card(
        "Final Approach",
        "Only shifted historical features were retained, ensuring that predictions are based strictly on past information.",
    )


# green success box to emphasise that fixing leakage was a key turning point
st.success(
    "Leakage prevention significantly improved the validity and credibility of the model results."
)


divider()


# overview of the three prediction tasks and which algorithm families were used for each
st.subheader("Modelling Approach")

model1, model2, model3 = st.columns(3)

# win prediction is a classification problem (home win = 1, away win = 0)
with model1:
    glass_card(
        "Win Prediction",
        "Classification models (Logistic Regression, Random Forest, Gradient Boosting) were used to predict whether the home team wins.",
    )

# score prediction is a regression problem (predicting a continuous number)
with model2:
    glass_card(
        "Score Prediction",
        "Regression models (Ridge, Gradient Boosting) were used to predict home score, away score, spread, and total points.",
    )

# player prediction is also regression but at the individual player level
with model3:
    glass_card(
        "Player Prediction",
        "Ridge regression was used to predict player points, rebounds, and assists using player-level features.",
    )


divider()


# describe how each model was judged — different metrics suit classification vs regression
st.subheader("Evaluation Strategy")

# two columns for the two families of metrics
e1, e2 = st.columns(2)

# classification metrics measure how well the model picks the right outcome (win/loss)
with e1:
    glass_card(
        "Classification Metrics",
        "Accuracy, Precision, Recall, F1 Score, and ROC-AUC were used to evaluate win prediction models.",
    )

# regression metrics measure how close the predicted numbers are to the actual values
with e2:
    glass_card(
        "Regression Metrics",
        "MAE, RMSE, and R² were used to evaluate score and player prediction models.",
    )

e3, e4 = st.columns(2)

# time-based splitting means the model is always tested on games it has never seen
# this is much more realistic than a random train/test split for time-series data
with e3:
    glass_card(
        "Time-Based Validation",
        "Models were trained on earlier games and tested on later games to simulate real-world prediction conditions.",
    )

# practical metrics check things that matter to a fan, not just a data scientist
with e4:
    glass_card(
        "Practical Metrics",
        "Additional metrics such as spread direction accuracy and within-margin accuracy were used for real-world relevance.",
    )


divider()


# show a simple text diagram of how data flows through the whole system end to end
st.subheader("System Architecture")

# st.code renders the text in a monospace block, making the arrow diagram easy to read
st.code(
    """
Raw NBA Data
    ↓
Data Cleaning & Preprocessing
    ↓
Feature Engineering (Rolling Stats)
    ↓
Pregame ML Datasets
    ↓
Model Training & Evaluation
    ↓
Saved Models & Results
    ↓
Streamlit Dashboard Interface
""",
    language="text",
)


divider()


# explain the reasoning behind the key design decisions in the project
st.subheader("Research Justification")

j1, j2, j3 = st.columns(3)

# machine learning is justified because NBA games involve many interacting factors
with j1:
    glass_card(
        "Why Machine Learning?",
        "NBA outcomes depend on many interacting variables, making machine learning suitable for capturing complex patterns.",
    )

# comparing multiple model types gives a clearer picture of what works and why
with j2:
    glass_card(
        "Why Multiple Models?",
        "Comparing linear and non-linear models allows evaluation of different types of relationships in the data.",
    )

# SHAP makes the model's decisions inspectable so we can trust what it's learned
with j3:
    glass_card(
        "Why Explainability?",
        "SHAP explainability ensures transparency and helps validate that the model uses meaningful basketball features.",
    )


# closing success banner summarising the scope of the project
st.success(
    "This methodology demonstrates a complete end-to-end machine learning system."
)
