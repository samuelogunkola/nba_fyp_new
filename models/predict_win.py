import pandas as pd
import numpy as np
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


# =========================
# LOAD DATA
# =========================
df = pd.read_parquet("data/processed/win_matchups.parquet")

print("Loaded win_matchups shape:", df.shape)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)


# =========================
# TARGET
# =========================
y = df["home_win"].copy()


# =========================
# FEATURES
# =========================
drop_cols = ["home_win", "gameid", "date", "home", "away"]

X = df.drop(columns=drop_cols, errors="ignore").copy()
X = X.select_dtypes(include=[np.number]).copy()
X = X.replace([np.inf, -np.inf], np.nan)

# 🚨 REMOVE LEAKAGE FEATURES
leakage_keywords = ["win"]
leakage_cols = [c for c in X.columns if any(k in c.lower() for k in leakage_keywords)]

print("Win-related columns dropped:", len(leakage_cols))
X = X.drop(columns=leakage_cols, errors="ignore")

print("Feature matrix shape:", X.shape)
print("Missing values before split:", int(X.isna().sum().sum()))


# =========================
# TIME-BASED SPLIT
# =========================
split_date = "2019-01-01"

train_idx = df["date"] < split_date
test_idx = df["date"] >= split_date

X_train = X.loc[train_idx].copy()
X_test = X.loc[test_idx].copy()
y_train = y.loc[train_idx].copy()
y_test = y.loc[test_idx].copy()

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)


# =========================
# DROP ALL-NAN COLUMNS
# =========================
all_nan_train_cols = X_train.columns[X_train.isna().all()].tolist()

if all_nan_train_cols:
    X_train = X_train.drop(columns=all_nan_train_cols)
    X_test = X_test.drop(columns=all_nan_train_cols)

print("Dropped all-NaN columns:", len(all_nan_train_cols))


# =========================
# IMPUTATION
# =========================
imputer = SimpleImputer(strategy="median")

X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test_imputed = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)


# =========================
# SCALING (for LR)
# =========================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)


# =========================
# MODEL 1: LOGISTIC REGRESSION
# =========================
lr = LogisticRegression(max_iter=2000, random_state=42)
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)

print("\n=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("F1:", f1_score(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))


# =========================
# MODEL 2: RANDOM FOREST (BALANCED)
# =========================
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=14,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_imputed, y_train)

y_pred_rf = rf.predict(X_test_imputed)

print("\n=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("F1:", f1_score(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))


# =========================
# MODEL 3: GRADIENT BOOSTING
# =========================
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train_imputed, y_train)

y_pred_gb = gb.predict(X_test_imputed)

print("\n=== Gradient Boosting ===")
print("Accuracy:", accuracy_score(y_test, y_pred_gb))
print("F1:", f1_score(y_test, y_pred_gb))
print(confusion_matrix(y_test, y_pred_gb))


# =========================
# FEATURE IMPORTANCE (RF)
# =========================
importances = pd.Series(rf.feature_importances_, index=X_train_imputed.columns)
top_features = importances.sort_values(ascending=False)

print("\nTop 20 Features:")
print(top_features.head(20))


# =========================
# FEATURE SELECTION EXPERIMENT
# =========================
top_n = 50
selected_features = top_features.head(top_n).index

X_train_sel = X_train_imputed[selected_features]
X_test_sel = X_test_imputed[selected_features]

rf.fit(X_train_sel, y_train)
y_pred_sel = rf.predict(X_test_sel)

print("\n=== Random Forest (Top 50 Features) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_sel))
print("F1:", f1_score(y_test, y_pred_sel))


# =========================
# SAVE MODELS
# =========================
joblib.dump(rf, "models/win_model.pkl")
joblib.dump(imputer, "models/win_imputer.pkl")
joblib.dump(scaler, "models/win_scaler.pkl")

print("\nSaved models.")