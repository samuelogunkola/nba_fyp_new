from pathlib import Path

import pandas as pd
import streamlit as st

from ui import apply_global_styles, render_sidebar, page_header, metric_card, glass_card, divider


st.set_page_config(
    page_title="Model Insights",
    page_icon="📉",
    layout="wide",
)

apply_global_styles()
render_sidebar()


WIN_RESULTS = Path("models/experiments/win/artifacts/win_results.csv")
SCORE_RESULTS = Path("models/experiments/score/artifacts/score_results.csv")
PLAYER_RESULTS = Path("models/experiments/player/artifacts/player_results_fast.csv")


@st.cache_data
def load_csv(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None


win_df = load_csv(WIN_RESULTS)
score_df = load_csv(SCORE_RESULTS)
player_df = load_csv(PLAYER_RESULTS)


page_header(
    "📉",
    "Model Insights",
    "Compare trained model performance across win prediction, score prediction, and player stat prediction.",
    ["Model Evaluation", "Experiment Results", "Performance Analysis"],
)

st.info(
    "These metrics come from the full local machine learning experiments. "
    "The hosted app uses lightweight demo datasets to keep the usability-test version fast and accessible."
)

# ============================================================
# TOP SUMMARY
# ============================================================

st.subheader("Overall Performance Summary")

c1, c2, c3, c4 = st.columns(4)

with c1:
    if win_df is not None:
        best_f1 = win_df.sort_values("f1", ascending=False).iloc[0]
        metric_card("Best Win Model", best_f1["model_name"], f"F1: {best_f1['f1']:.3f}")
    else:
        metric_card("Best Win Model", "Missing", "Run win experiment")

with c2:
    if score_df is not None:
        ridge_home = score_df[(score_df["model_name"] == "Ridge") & (score_df["target"] == "home_pts")]
        if not ridge_home.empty:
            metric_card("Best Score Model", "Ridge", f"Home MAE: {ridge_home.iloc[0]['mae']:.2f}")
        else:
            metric_card("Best Score Model", "N/A", "No Ridge result")
    else:
        metric_card("Best Score Model", "Missing", "Run score experiment")

with c3:
    if player_df is not None:
        pts = player_df[player_df["target"] == "pts"].iloc[0]
        metric_card("Best Player Model", "Ridge", f"PTS R²: {pts['r2']:.3f}")
    else:
        metric_card("Best Player Model", "Missing", "Run player experiment")

with c4:
    metric_card("Evaluation Type", "Time Split", "Pregame leakage-safe testing")


divider()


# ============================================================
# WIN MODEL SECTION
# ============================================================

st.subheader("🏆 Win Prediction Models")

if win_df is None:
    st.warning(f"Missing file: `{WIN_RESULTS}`")
else:
    best_accuracy = win_df.sort_values("accuracy", ascending=False).iloc[0]
    best_f1 = win_df.sort_values("f1", ascending=False).iloc[0]
    best_auc = win_df.sort_values("roc_auc", ascending=False).iloc[0]

    w1, w2, w3 = st.columns(3)

    with w1:
        metric_card("Best Accuracy", best_accuracy["model_name"], f"{best_accuracy['accuracy']:.3f}")

    with w2:
        metric_card("Best F1", best_f1["model_name"], f"{best_f1['f1']:.3f}")

    with w3:
        metric_card("Best ROC-AUC", best_auc["model_name"], f"{best_auc['roc_auc']:.3f}")

    st.markdown("#### Win Model Comparison")

    chart_df = win_df.set_index("model_name")[["accuracy", "precision", "recall", "f1", "roc_auc"]]
    st.bar_chart(chart_df)

    st.dataframe(win_df, use_container_width=True)

    glass_card(
        "Interpretation",
        "Gradient Boosting achieved the strongest F1 score, making it the best model when balancing precision and recall. Random Forest achieved similar accuracy, while Logistic Regression provided a useful linear baseline.",
    )


divider()


# ============================================================
# SCORE MODEL SECTION
# ============================================================

st.subheader("📈 Score Prediction Models")

if score_df is None:
    st.warning(f"Missing file: `{SCORE_RESULTS}`")
else:
    ridge_df = score_df[score_df["model_name"] == "Ridge"].copy()

    if not ridge_df.empty:
        s1, s2, s3, s4 = st.columns(4)

        def get_ridge_metric(target, metric):
            row = ridge_df[ridge_df["target"] == target]
            if row.empty:
                return None
            return row.iloc[0][metric]

        home_mae = get_ridge_metric("home_pts", "mae")
        away_mae = get_ridge_metric("away_pts", "mae")
        spread_mae = get_ridge_metric("point_spread", "mae")
        total_mae = get_ridge_metric("total_points", "mae")

        with s1:
            metric_card("Home MAE", f"{home_mae:.2f}" if home_mae is not None else "N/A", "Ridge")

        with s2:
            metric_card("Away MAE", f"{away_mae:.2f}" if away_mae is not None else "N/A", "Ridge")

        with s3:
            metric_card("Spread MAE", f"{spread_mae:.2f}" if spread_mae is not None else "N/A", "Ridge")

        with s4:
            metric_card("Total MAE", f"{total_mae:.2f}" if total_mae is not None else "N/A", "Ridge")

    st.markdown("#### Score Model MAE Comparison")

    mae_chart = score_df.pivot_table(
        index="target",
        columns="model_name",
        values="mae",
        aggfunc="mean",
    )

    st.bar_chart(mae_chart)

    st.dataframe(score_df, use_container_width=True)

    glass_card(
        "Interpretation",
        "Score prediction was the most difficult regression task. Ridge Regression performed best overall, suggesting that the rolling pregame features contain mostly linear signal. The low R² values are realistic because exact NBA scores are highly variable.",
    )


divider()


# ============================================================
# PLAYER MODEL SECTION
# ============================================================

st.subheader("👤 Player Stat Prediction Models")

if player_df is None:
    st.warning(f"Missing file: `{PLAYER_RESULTS}`")
else:
    p1, p2, p3 = st.columns(3)

    pts = player_df[player_df["target"] == "pts"].iloc[0]
    reb = player_df[player_df["target"] == "reb"].iloc[0]
    ast = player_df[player_df["target"] == "ast"].iloc[0]

    with p1:
        metric_card("Points Model", f"MAE {pts['mae']:.2f}", f"R²: {pts['r2']:.3f}")

    with p2:
        metric_card("Rebounds Model", f"MAE {reb['mae']:.2f}", f"R²: {reb['r2']:.3f}")

    with p3:
        metric_card("Assists Model", f"MAE {ast['mae']:.2f}", f"R²: {ast['r2']:.3f}")

    st.markdown("#### Player Stat R² Comparison")

    r2_chart = player_df.set_index("target")[["r2"]]
    st.bar_chart(r2_chart)

    st.markdown("#### Player Stat MAE Comparison")

    mae_chart = player_df.set_index("target")[["mae"]]
    st.bar_chart(mae_chart)

    st.dataframe(player_df, use_container_width=True)

    glass_card(
        "Interpretation",
        "Player stat prediction produced the strongest regression results. Points achieved the highest R², while rebounds and assists achieved very low average errors. This suggests recent player-level rolling features provide strong predictive signal.",
    )


divider()


# ============================================================
# ACADEMIC SUMMARY
# ============================================================

st.subheader("Academic Evaluation Summary")

a1, a2 = st.columns(2)

with a1:
    glass_card(
        "Model Difficulty",
        "The results show that prediction difficulty varies by task. Exact score prediction is hardest because basketball scoring has high variance, while player stat prediction benefits from stable individual historical patterns.",
    )

with a2:
    glass_card(
        "Model Selection",
        "Ridge Regression was effective for regression tasks because rolling averages and efficiency metrics provide mainly linear signal. Gradient Boosting performed best for win classification due to its ability to capture non-linear matchup patterns.",
    )

