import os
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from models.core.preprocess import clean_features, Preprocessor, drop_all_nan_train_columns
from models.core.split import time_split


# =========================
# CONFIG
# =========================
DATA_PATH = "data/processed/win_matchups.parquet"
OUTPUT_DIR = "models/experiments/win/shap_outputs"
SPLIT_DATE = "2019-01-01"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# LOAD DATA
# =========================
df = pd.read_parquet(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

print("Loaded win_matchups:", df.shape)


# =========================
# TARGET / FEATURES
# =========================
target_col = "home_win"

drop_cols = [
    "home_win",
    "gameid",
    "date",
    "home",
    "away",
]

X = clean_features(
    df,
    drop_cols=drop_cols,
    remove_win_leakage=True
)
y = df[target_col].copy()

print("Feature matrix shape:", X.shape)


# =========================
# TIME SPLIT
# =========================
train_idx, test_idx = time_split(df, date_col="date", split_date=SPLIT_DATE)

X_train = X.loc[train_idx].copy()
X_test = X.loc[test_idx].copy()
y_train = y.loc[train_idx].copy()
y_test = y.loc[test_idx].copy()

X_train, X_test, dropped_cols = drop_all_nan_train_columns(X_train, X_test)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)
print("Dropped all-NaN cols:", len(dropped_cols))


# =========================
# PREPROCESS
# =========================
preprocessor = Preprocessor(scale=False)

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

X_train_proc = pd.DataFrame(X_train_proc, columns=X_train.columns, index=X_train.index)
X_test_proc = pd.DataFrame(X_test_proc, columns=X_test.columns, index=X_test.index)


# =========================
# MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=14,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train_proc, y_train)

joblib.dump(
    {
        "model": model,
        "preprocessor": preprocessor,
        "feature_columns": X_train.columns.tolist(),
    },
    os.path.join(OUTPUT_DIR, "win_rf_shap_bundle.pkl")
)


# =========================
# SAMPLE TEST SET
# =========================
sample_size = min(1000, len(X_test_proc))
X_sample = X_test_proc.sample(n=sample_size, random_state=42)
y_sample = y_test.loc[X_sample.index]

print("SHAP sample size:", X_sample.shape)


# =========================
# SHAP EXPLAINER
# =========================
explainer = shap.TreeExplainer(model)
shap_values_raw = explainer.shap_values(X_sample)

# Force SHAP values into shape: (n_samples, n_features) for class 1
if isinstance(shap_values_raw, list):
    # Old SHAP style: list[class0, class1]
    shap_values = shap_values_raw[1]
else:
    shap_values = np.array(shap_values_raw)

    # Common problematic case: (n_samples, n_features, 2)
    if shap_values.ndim == 3 and shap_values.shape[2] == 2:
        shap_values = shap_values[:, :, 1]
    # Less common case: (n_samples, 2, n_features)
    elif shap_values.ndim == 3 and shap_values.shape[1] == 2:
        shap_values = shap_values[:, 1, :]
    # Already correct case: (n_samples, n_features)
    elif shap_values.ndim == 2:
        pass
    else:
        raise ValueError(f"Unexpected SHAP values shape: {shap_values.shape}")

print("Final SHAP matrix shape:", shap_values.shape)


# =========================
# GLOBAL SUMMARY PLOT
# =========================
plt.figure()
shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
plt.tight_layout()
summary_path = os.path.join(OUTPUT_DIR, "win_shap_summary.png")
plt.savefig(summary_path, dpi=200, bbox_inches="tight")
plt.close()

print("Saved:", summary_path)


# =========================
# BAR IMPORTANCE PLOT
# =========================
plt.figure()
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=20)
plt.tight_layout()
bar_path = os.path.join(OUTPUT_DIR, "win_shap_bar.png")
plt.savefig(bar_path, dpi=200, bbox_inches="tight")
plt.close()

print("Saved:", bar_path)


# =========================
# LOCAL WATERFALL EXAMPLE
# =========================
example_idx = 0

expected_value = explainer.expected_value
if isinstance(expected_value, list):
    expected_value = expected_value[1]
else:
    expected_value = np.array(expected_value)
    if expected_value.ndim == 1 and len(expected_value) == 2:
        expected_value = expected_value[1]
    elif expected_value.ndim == 0:
        expected_value = float(expected_value)

example_values = shap_values[example_idx]

explanation = shap.Explanation(
    values=example_values,
    base_values=expected_value,
    data=X_sample.iloc[example_idx].values,
    feature_names=X_sample.columns.tolist()
)

plt.figure()
shap.plots.waterfall(explanation, max_display=15, show=False)
plt.tight_layout()
waterfall_path = os.path.join(OUTPUT_DIR, "win_shap_waterfall_example.png")
plt.savefig(waterfall_path, dpi=200, bbox_inches="tight")
plt.close()

print("Saved:", waterfall_path)


# =========================
# TOP FEATURES TABLE
# =========================
mean_abs_shap = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=X_sample.columns
).sort_values(ascending=False)

top20 = mean_abs_shap.head(20)
top20_path = os.path.join(OUTPUT_DIR, "win_shap_top20.csv")
top20.to_csv(top20_path, header=["mean_abs_shap"])

print("\nTop 20 SHAP features:")
print(top20)
print("\nSaved:", top20_path)


# =========================
# EXAMPLE PREDICTIONS
# =========================
proba = model.predict_proba(X_sample)[:, 1]
pred = model.predict(X_sample)

example_table = pd.DataFrame({
    "actual_home_win": y_sample.values,
    "pred_home_win": pred,
    "pred_home_win_proba": proba
}, index=X_sample.index)

example_table_path = os.path.join(OUTPUT_DIR, "win_shap_examples.csv")
example_table.to_csv(example_table_path)

print("Saved:", example_table_path)
print("\nDone.")