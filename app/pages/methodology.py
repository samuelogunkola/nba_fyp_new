import streamlit as st

from ui import apply_global_styles, render_sidebar, page_header, metric_card, glass_card, divider


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Methodology",
    page_icon="🧪",
    layout="wide",
)

apply_global_styles()
render_sidebar()


# ============================================================
# HEADER
# ============================================================

page_header(
    "🧪",
    "Methodology",
    "A structured overview of the full machine learning pipeline, including data preparation, feature engineering, modelling, leakage prevention, and evaluation.",
    ["System Design", "ML Pipeline", "Leakage-Safe", "Evaluation"],
)


# ============================================================
# SUMMARY METRICS
# ============================================================

st.subheader("Methodology Summary")

m1, m2, m3, m4 = st.columns(4)

with m1:
    metric_card("Data Source", "NBA Historical Data", "Player & team box scores")

with m2:
    metric_card("Validation", "Time-Based Split", "Past → Train, Future → Test")

with m3:
    metric_card("Feature Type", "Pregame Only", "Rolling historical features")

with m4:
    metric_card("System Outputs", "3 Models", "Win, Score, Player Stats")


divider()


# ============================================================
# DATA PIPELINE
# ============================================================

st.subheader("End-to-End Data Pipeline")

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
        "All models and results were integrated into an interactive Streamlit application for real-time predictions.",
    )


divider()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

st.subheader("Feature Engineering Strategy")

f1, f2, f3 = st.columns(3)

with f1:
    glass_card(
        "Rolling Averages",
        "3, 5, and 10-game rolling averages were used to represent recent performance trends.",
    )

with f2:
    glass_card(
        "Efficiency Metrics",
        "Derived metrics such as shooting percentages, efficiency ratings, and usage proxies were included.",
    )

with f3:
    glass_card(
        "Matchup Features",
        "Differences between home and away teams were calculated to represent relative team strength.",
    )


st.info(
    "Only pregame features were used to ensure that predictions are realistic and usable before a game takes place."
)


divider()


# ============================================================
# LEAKAGE PREVENTION
# ============================================================

st.subheader("Leakage Prevention")

l1, l2, l3 = st.columns(3)

with l1:
    glass_card(
        "Initial Issue",
        "Early models showed unrealistically high accuracy, indicating that future (post-game) information was leaking into the features.",
    )

with l2:
    glass_card(
        "Leakage Removal",
        "All same-game statistics such as final scores, win labels, and plus-minus values were removed from the feature set.",
    )

with l3:
    glass_card(
        "Final Approach",
        "Only shifted historical features were retained, ensuring that predictions are based strictly on past information.",
    )


st.success(
    "Leakage prevention significantly improved the validity and credibility of the model results."
)


divider()


# ============================================================
# MODELLING
# ============================================================

st.subheader("Modelling Approach")

model1, model2, model3 = st.columns(3)

with model1:
    glass_card(
        "Win Prediction",
        "Classification models (Logistic Regression, Random Forest, Gradient Boosting) were used to predict whether the home team wins.",
    )

with model2:
    glass_card(
        "Score Prediction",
        "Regression models (Ridge, Gradient Boosting) were used to predict home score, away score, spread, and total points.",
    )

with model3:
    glass_card(
        "Player Prediction",
        "Ridge regression was used to predict player points, rebounds, and assists using player-level features.",
    )


divider()


# ============================================================
# EVALUATION
# ============================================================

st.subheader("Evaluation Strategy")

e1, e2 = st.columns(2)

with e1:
    glass_card(
        "Classification Metrics",
        "Accuracy, Precision, Recall, F1 Score, and ROC-AUC were used to evaluate win prediction models.",
    )

with e2:
    glass_card(
        "Regression Metrics",
        "MAE, RMSE, and R² were used to evaluate score and player prediction models.",
    )

e3, e4 = st.columns(2)

with e3:
    glass_card(
        "Time-Based Validation",
        "Models were trained on earlier games and tested on later games to simulate real-world prediction conditions.",
    )

with e4:
    glass_card(
        "Practical Metrics",
        "Additional metrics such as spread direction accuracy and within-margin accuracy were used for real-world relevance.",
    )


divider()


# ============================================================
# SYSTEM ARCHITECTURE
# ============================================================

st.subheader("System Architecture")

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


# ============================================================
# RESEARCH JUSTIFICATION
# ============================================================

st.subheader("Research Justification")

j1, j2, j3 = st.columns(3)

with j1:
    glass_card(
        "Why Machine Learning?",
        "NBA outcomes depend on many interacting variables, making machine learning suitable for capturing complex patterns.",
    )

with j2:
    glass_card(
        "Why Multiple Models?",
        "Comparing linear and non-linear models allows evaluation of different types of relationships in the data.",
    )

with j3:
    glass_card(
        "Why Explainability?",
        "SHAP explainability ensures transparency and helps validate that the model uses meaningful basketball features.",
    )


st.success(
    "This methodology demonstrates a complete end-to-end machine learning system, suitable for a high-grade final-year project."
)