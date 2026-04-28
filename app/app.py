# streamlit is the framework that powers the entire web app
import streamlit as st

# shared UI helpers used on every page for consistent styling and layout
from ui import apply_global_styles, render_sidebar, page_header, metric_card, glass_card, divider


# this is the homepage of the app — it's the first page the user sees when they open it
# set_page_config must be the very first streamlit call in the script, before anything else
st.set_page_config(
   page_title="NBA Analytics Platform",
    page_icon="🏀",
    layout="wide",               # stretch content across the full browser width
    initial_sidebar_state="expanded"  # open the sidebar by default so navigation is visible
)

# apply the shared CSS styles so the homepage looks the same as all the other pages
apply_global_styles()

# draw the left-hand sidebar navigation that lets users move between pages
render_sidebar()


# render the top banner with the platform name, description, and tag pills
page_header(
    "🏀",
    "NBA Analytics Platform",
    "A machine learning dashboard for NBA win prediction, score forecasting, player stat prediction, model evaluation, and explainable AI.",
    ["Final Year Project", "Machine Learning", "Pregame Prediction", "Explainable AI"],
)

# let usability testers know they're working with lightweight demo data, not the full dataset
st.info(
    "Hosted demo note: this online version uses lightweight demo datasets so participants can access the app reliably during usability testing."
)


# four headline numbers at the top of the homepage so visitors immediately understand
# what the system achieved — these come from the full local model experiments
col1, col2, col3, col4 = st.columns(4)

with col1:
    # F1 score of the best win prediction model — balances precision and recall
    metric_card("Win Model", "F1 0.699", "Gradient Boosting (best classification)")

with col2:
    # MAE (mean absolute error) for score prediction — off by about 9.5 points on average
    metric_card("Score Model", "MAE 9.54", "Ridge regression")

with col3:
    # R² for the player points model — explains ~64% of the variation in actual points scored
    metric_card("Player Points", "R² 0.642", "Strongest model performance")

with col4:
    # total number of player-game records used across all experiments
    metric_card("Dataset Size", "700k+", "Player-game records")


# horizontal line to separate the headline metrics from the feature cards below
divider()


# three cards giving a quick overview of what the platform can do
left, mid, right = st.columns(3)

with left:
    glass_card(
        "Prediction Tools",
        "Use trained models to estimate match outcomes, projected scores, and individual player statistics using historical pregame features.",
    )

with mid:
    glass_card(
        "Research-Grade Evaluation",
        "The system evaluates models using accuracy, F1 score, ROC-AUC, MAE, RMSE, R², and within-margin performance metrics.",
    )

with right:
    glass_card(
        "Explainability",
        "SHAP visualisations explain which features most influence predictions, improving transparency and interpretability.",
    )


divider()


# two-column layout listing what the platform can do in bullet point form
# left column focuses on what a user can actually interact with
# right column covers the research and engineering decisions behind the system
col1, col2 = st.columns(2)

with col1:
    st.subheader("Platform Capabilities")

    # each bullet describes a feature the user can explore via the sidebar navigation
    st.markdown("""
- Win prediction (home team probability)
- Score prediction (home and away points)
- Player stat prediction (PTS, REB, AST)
- Model comparison and insights
- Explainability using SHAP
- Full ML pipeline from raw data to deployment
""")

with col2:
    st.subheader("Research Features")

    # each bullet describes a technical or academic design decision in the project
    st.markdown("""
- Leakage-safe pregame features
- Time-based train/test evaluation
- Feature engineering with rolling statistics
- Model benchmarking (Linear vs Tree-based)
- SHAP explainability (global + local)
- Saved model artifacts for inference
""")


divider()


# simple call-to-action at the bottom pointing the user to the sidebar
st.success("Use the sidebar to explore the platform.")
