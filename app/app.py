import streamlit as st

from ui import apply_global_styles, render_sidebar, page_header, metric_card, glass_card, divider


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
   page_title="NBA Analytics Platform",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_global_styles()
render_sidebar()


# ============================================================
# HOMEPAGE
# ============================================================

page_header(
    "🏀",
    "NBA Analytics Platform",
    "A machine learning dashboard for NBA win prediction, score forecasting, player stat prediction, model evaluation, and explainable AI.",
    ["Final Year Project", "Machine Learning", "Pregame Prediction", "Explainable AI"],
)


# =========================
# TOP METRICS
# =========================

col1, col2, col3, col4 = st.columns(4)

with col1:
    metric_card("Win Model", "F1 0.699", "Gradient Boosting (best classification)")

with col2:
    metric_card("Score Model", "MAE 9.54", "Ridge regression")

with col3:
    metric_card("Player Points", "R² 0.642", "Strongest model performance")

with col4:
    metric_card("Dataset Size", "700k+", "Player-game records")


divider()


# =========================
# FEATURE CARDS
# =========================

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


# =========================
# CAPABILITIES + FEATURES
# =========================

col1, col2 = st.columns(2)

with col1:
    st.subheader("Platform Capabilities")

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

    st.markdown("""
- Leakage-safe pregame features  
- Time-based train/test evaluation  
- Feature engineering with rolling statistics  
- Model benchmarking (Linear vs Tree-based)  
- SHAP explainability (global + local)  
- Saved model artifacts for inference  
""")


divider()



# =========================
# FINAL CTA
# =========================

st.success("Use the sidebar to explore the platform.")