from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[3]

DATA_PATH = BASE_DIR / "data" / "processed" / "win_matchups.parquet"
MODEL_PATH = BASE_DIR / "models" / "experiments" / "win" / "artifacts" / "logistic_regression_home_win.pkl"

OUTPUT_DIR = BASE_DIR / "models" / "experiments" / "win" / "report_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


df = pd.read_parquet(DATA_PATH)

target_col = "home_win"

# Keep only numeric columns and remove obvious target/leakage columns
drop_cols = [
    target_col,
    "home_pts",
    "away_pts",
    "home_score",
    "away_score",
    "home_plus_minus",
    "away_plus_minus",
]

X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
X = X.select_dtypes(include=["number"]).fillna(0)

y = df[target_col]

# Same style of chronological split used for realistic evaluation
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

model = joblib.load(MODEL_PATH)

# If model is a pipeline, this still works
y_prob = model.predict_proba(X_test)[:, 1]


# ---------------------------------------------------------
# 1. Calibration curve
# ---------------------------------------------------------
prob_true, prob_pred = calibration_curve(
    y_test,
    y_prob,
    n_bins=10,
    strategy="uniform"
)

brier = brier_score_loss(y_test, y_prob)

plt.figure(figsize=(7, 6))
plt.plot(prob_pred, prob_true, marker="o", label="Model calibration")
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
plt.xlabel("Mean predicted probability")
plt.ylabel("Observed win frequency")
plt.title(f"Calibration Curve for Home Win Prediction\nBrier Score: {brier:.3f}")
plt.legend()
plt.tight_layout()

calibration_path = OUTPUT_DIR / "calibration_curve.png"
plt.savefig(calibration_path, dpi=300)
plt.close()


# ---------------------------------------------------------
# 2. Feature importance
# ---------------------------------------------------------
result = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=5,
    random_state=42,
    scoring="roc_auc",
    n_jobs=-1
)

importance_df = pd.DataFrame({
    "feature": X_test.columns,
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std
}).sort_values("importance_mean", ascending=False)

importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

top_features = importance_df.head(15).sort_values("importance_mean")

plt.figure(figsize=(9, 6))
plt.barh(top_features["feature"], top_features["importance_mean"])
plt.xlabel("Mean decrease in ROC-AUC")
plt.ylabel("Feature")
plt.title("Top 15 Permutation Feature Importances")
plt.tight_layout()

importance_path = OUTPUT_DIR / "feature_importance_bar_chart.png"
plt.savefig(importance_path, dpi=300)
plt.close()

print("Saved report figures:")
print(calibration_path)
print(importance_path)
print(OUTPUT_DIR / "feature_importance.csv")