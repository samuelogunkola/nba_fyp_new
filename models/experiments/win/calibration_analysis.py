import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

from models.core.preprocess import clean_features, Preprocessor, drop_all_nan_train_columns
from models.core.split import time_split


# =========================
# LOAD DATA
# =========================
df = pd.read_parquet("data/processed/win_matchups.parquet")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)


# =========================
# PREP
# =========================
target_col = "home_win"

drop_cols = [
    "home_win",
    "gameid",
    "date",
    "home",
    "away",
]

X = clean_features(df, drop_cols=drop_cols)
y = df[target_col]

train_idx, test_idx = time_split(df, split_date="2019-01-01")

X_train = X.loc[train_idx]
X_test = X.loc[test_idx]
y_train = y.loc[train_idx]
y_test = y.loc[test_idx]

X_train, X_test, _ = drop_all_nan_train_columns(X_train, X_test)

preprocessor = Preprocessor(scale=True)
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


# =========================
# MODEL
# =========================
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]


# =========================
# CALIBRATION CURVE
# =========================
true_prob, pred_prob = calibration_curve(y_test, probs, n_bins=10)


# =========================
# PLOT
# =========================
plt.figure(figsize=(6, 6))
plt.plot(pred_prob, true_prob, marker='o', label="Model")
plt.plot([0, 1], [0, 1], linestyle='--', label="Perfectly calibrated")

plt.xlabel("Predicted probability")
plt.ylabel("Actual probability")
plt.title("Calibration Curve (Win Prediction)")
plt.legend()
plt.grid()

plt.show()