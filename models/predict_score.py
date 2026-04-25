import pandas as pd
import numpy as np
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================
# LOAD DATA
# =========================
df = pd.read_parquet("data/processed/score_matchups.parquet")

print("Loaded score_matchups shape:", df.shape)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)


# =========================
# TARGETS
# =========================
y_home = df["home_pts"].copy()
y_away = df["away_pts"].copy()


# =========================
# FEATURES
# =========================
drop_cols = [
    "home_pts",
    "away_pts",
    "gameid",
    "date",
    "home",
    "away",
]

X = df.drop(columns=drop_cols, errors="ignore").copy()
X = X.select_dtypes(include=[np.number]).copy()
X = X.replace([np.inf, -np.inf], np.nan)

# Remove any win-related leakage (just in case)
leakage_cols = [c for c in X.columns if "win" in c.lower()]
X = X.drop(columns=leakage_cols, errors="ignore")

print("Feature matrix shape:", X.shape)
print("Missing values:", int(X.isna().sum().sum()))


# =========================
# TIME SPLIT
# =========================
split_date = "2019-01-01"

train_idx = df["date"] < split_date
test_idx = df["date"] >= split_date

X_train = X.loc[train_idx].copy()
X_test = X.loc[test_idx].copy()

y_home_train = y_home.loc[train_idx]
y_home_test = y_home.loc[test_idx]

y_away_train = y_away.loc[train_idx]
y_away_test = y_away.loc[test_idx]

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)


# =========================
# IMPUTATION
# =========================
imputer = SimpleImputer(strategy="median")

X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)


# =========================
# SCALING (for linear models)
# =========================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================
# FUNCTION: EVALUATE
# =========================
def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n{name}")
    print("MAE:", mae)
    print("RMSE:", rmse)


# =========================
# MODEL 1: RIDGE (baseline)
# =========================
ridge_home = Ridge()
ridge_home.fit(X_train_scaled, y_home_train)

pred_home_ridge = ridge_home.predict(X_test_scaled)
evaluate(y_home_test, pred_home_ridge, "Ridge (Home Points)")

ridge_away = Ridge()
ridge_away.fit(X_train_scaled, y_away_train)

pred_away_ridge = ridge_away.predict(X_test_scaled)
evaluate(y_away_test, pred_away_ridge, "Ridge (Away Points)")


# =========================
# MODEL 2: RANDOM FOREST
# =========================
rf_home = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

rf_home.fit(X_train, y_home_train)
pred_home_rf = rf_home.predict(X_test)
evaluate(y_home_test, pred_home_rf, "Random Forest (Home Points)")

rf_away = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

rf_away.fit(X_train, y_away_train)
pred_away_rf = rf_away.predict(X_test)
evaluate(y_away_test, pred_away_rf, "Random Forest (Away Points)")


# =========================
# MODEL 3: GRADIENT BOOSTING
# =========================
gb_home = GradientBoostingRegressor(random_state=42)
gb_home.fit(X_train, y_home_train)

pred_home_gb = gb_home.predict(X_test)
evaluate(y_home_test, pred_home_gb, "Gradient Boosting (Home Points)")

gb_away = GradientBoostingRegressor(random_state=42)
gb_away.fit(X_train, y_away_train)

pred_away_gb = gb_away.predict(X_test)
evaluate(y_away_test, pred_away_gb, "Gradient Boosting (Away Points)")


# =========================
# SAVE MODELS
# =========================
joblib.dump(rf_home, "models/home_score_model.pkl")
joblib.dump(rf_away, "models/away_score_model.pkl")

print("\nSaved score models.")