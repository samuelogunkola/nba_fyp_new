from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from ui import apply_global_styles, render_sidebar, page_header, metric_card, glass_card, divider


st.set_page_config(page_title="Explainability", page_icon="🔍", layout="wide")

apply_global_styles()
render_sidebar()


SHAP_DIR = Path("models/experiments/win/shap_outputs")
WIN_RESULTS_PATH = Path("models/experiments/win/artifacts/win_results.csv")


@st.cache_data
def load_csv(path: Path):
    if path and path.exists():
        return pd.read_csv(path)
    return None


def find_first_existing(patterns):
    for pattern in patterns:
        matches = list(SHAP_DIR.glob(pattern))
        if matches:
            return matches[0]
    return None


def safe_feature_chart(df, feature_col, value_col):
    chart_df = df.copy()
    chart_df[feature_col] = chart_df[feature_col].astype(str)
    chart_df[value_col] = pd.to_numeric(chart_df[value_col], errors="coerce")
    chart_df = chart_df.dropna(subset=[value_col]).sort_values(value_col)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(chart_df[feature_col], chart_df[value_col])
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("Feature")
    ax.set_title("Top SHAP Feature Importance")
    st.pyplot(fig)


win_results = load_csv(WIN_RESULTS_PATH)

shap_csv_path = find_first_existing([
    "*top20*.csv",
    "*summary*.csv",
    "*shap*.csv",
])

summary_img_path = find_first_existing([
    "*summary*.png",
])

importance_img_path = find_first_existing([
    "*bar*.png",
    "*importance*.png",
])

waterfall_img_path = find_first_existing([
    "*waterfall*.png",
    "*example*.png",
])

shap_df = load_csv(shap_csv_path)


page_header(
    "🔍",
    "Explainability",
    "Understand which features influence the NBA win prediction model using SHAP-based interpretation.",
    ["SHAP", "Feature Importance", "Transparency", "Win Prediction"],
)


st.subheader("Explainability Overview")

c1, c2, c3, c4 = st.columns(4)

with c1:
    metric_card("Method", "SHAP", "Feature contribution analysis")

with c2:
    metric_card("Model Explained", "Win Predictor", "Tree-based classifier")

with c3:
    if win_results is not None:
        gb = win_results[win_results["model_name"] == "Gradient Boosting"]
        if not gb.empty:
            metric_card("Model F1", f"{gb.iloc[0]['f1']:.3f}", "Best F1 score")
        else:
            metric_card("Model F1", "N/A", "Result unavailable")
    else:
        metric_card("Model F1", "N/A", "Win results missing")

with c4:
    if shap_df is not None:
        metric_card("SHAP Output", "Found", shap_csv_path.name)
    else:
        metric_card("SHAP Output", "Missing", "Check shap_outputs folder")


divider()


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

    st.stop()


# ============================================================
# SHAP TABLE HANDLING
# ============================================================

shap_df.columns = [str(c).strip() for c in shap_df.columns]

possible_feature_cols = [
    "feature",
    "Feature",
    "feature_name",
    "column",
    "Unnamed: 0",
]

possible_value_cols = [
    "mean_abs_shap",
    "mean_abs_SHAP",
    "importance",
    "Importance",
    "mean_shap",
    "shap_value",
    "SHAP",
]

feature_col = next((c for c in possible_feature_cols if c in shap_df.columns), None)
value_col = next((c for c in possible_value_cols if c in shap_df.columns), None)

if feature_col is None:
    feature_col = shap_df.columns[0]

if value_col is None:
    numeric_cols = shap_df.select_dtypes(include="number").columns.tolist()
    value_col = numeric_cols[-1] if numeric_cols else None


# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================

st.subheader("Global Feature Importance")

if value_col is None:
    st.warning("SHAP CSV found, but no numeric importance column could be detected.")
    st.dataframe(shap_df.head(50), use_column_width=True)
else:
    shap_df[value_col] = pd.to_numeric(shap_df[value_col], errors="coerce")
    shap_df = shap_df.dropna(subset=[value_col])
    shap_df = shap_df.sort_values(value_col, ascending=False).reset_index(drop=True)

    max_features = min(30, len(shap_df))

    top_n = st.slider(
        "Number of features to display",
        min_value=5,
        max_value=max_features,
        value=min(15, max_features),
    )

    top_df = shap_df[[feature_col, value_col]].head(top_n)

    safe_feature_chart(top_df, feature_col, value_col)

    st.dataframe(top_df, use_container_width=True)

    top_feature = str(top_df.iloc[0][feature_col])
    top_value = float(top_df.iloc[0][value_col])

    t1, t2, t3 = st.columns(3)

    with t1:
        metric_card("Top Feature", top_feature, "Most influential feature")

    with t2:
        metric_card("Top Importance", f"{top_value:.4f}", "Mean absolute SHAP value")

    with t3:
        metric_card("Features Displayed", str(top_n), "Current view")


divider()


# ============================================================
# SHAP IMAGES
# ============================================================

st.subheader("SHAP Visualisations")

img1, img2 = st.columns(2)

with img1:
    st.markdown("#### Summary Plot")
    if summary_img_path and summary_img_path.exists():
        st.image(str(summary_img_path), use_column_width=True)
    else:
        st.info("Summary plot image not found.")

with img2:
    st.markdown("#### Global Importance")
    if importance_img_path and importance_img_path.exists():
        st.image(str(importance_img_path), use_column_width=True)
    else:
        st.info("Global importance image not found.")


if waterfall_img_path and waterfall_img_path.exists():
    divider()
    st.subheader("Single Prediction Explanation")
    st.image(str(waterfall_img_path), use_column_width=True)


divider()


# ============================================================
# INTERPRETATION
# ============================================================

st.subheader("Interpretation")

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


st.success(
    "SHAP explainability strengthens the system by making model reasoning transparent and easier to evaluate."
)