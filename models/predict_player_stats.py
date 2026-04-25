import pandas as pd
import numpy as np
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================
# LOAD DATA
# =========================
df = pd.read_parquet("data/processed/player_stats_dataset.parquet")

print("Loaded player_stats_dataset shape:", df.shape)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)


# =========================
# TARGETS
# =========================
y_pts = df["pts"]
y_ast = df["ast"]
y_reb = df["reb"]


# =========================
# FEATURES
# =========================
drop_cols = ["pts", "ast", "reb", "gameid", "date", "player", "home", "away"]

X = df.drop(columns=drop_cols, errors="ignore")
X = X.select_dtypes(include=[np.number])
X = X.replace([np.inf, -np.inf], np.nan)

# remove leakage
leakage_cols = [c for c in X.columns if "win" in c.lower()]
X = X.drop(columns=leakage_cols, errors="ignore")

print("Feature matrix shape:", X.shape)


# =========================
# TIME SPLIT
# =========================
split_date = "2019-01-01"

train_idx = df["date"] < split_date
test_idx = df["date"] >= split_date

X_train = X.loc[train_idx]
X_test = X.loc[test_idx]

y_pts_train = y_pts.loc[train_idx]
y_pts_test = y_pts.loc[test_idx]

y_ast_train = y_ast.loc[train_idx]
y_ast_test = y_ast.loc[test_idx]

y_reb_train = y_reb.loc[train_idx]
y_reb_test = y_reb.loc[test_idx]

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)


# =========================
# IMPUTATION
# =========================
imputer = SimpleImputer(strategy="median")

X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)


# =========================
# SCALING
# =========================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================
# EVALUATION
# =========================
def eval_model(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n{name}")
    print("MAE:", mae)
    print("RMSE:", rmse)


# =========================
# RIDGE
# =========================
ridge_pts = Ridge()
ridge_pts.fit(X_train_scaled, y_pts_train)
eval_model(y_pts_test, ridge_pts.predict(X_test_scaled), "Ridge (PTS)")

ridge_ast = Ridge()
ridge_ast.fit(X_train_scaled, y_ast_train)
eval_model(y_ast_test, ridge_ast.predict(X_test_scaled), "Ridge (AST)")

ridge_reb = Ridge()
ridge_reb.fit(X_train_scaled, y_reb_train)
eval_model(y_reb_test, ridge_reb.predict(X_test_scaled), "Ridge (REB)")


# =========================
# ELASTIC NET
# =========================
enet_pts = ElasticNet(alpha=0.1, l1_ratio=0.5)
enet_pts.fit(X_train_scaled, y_pts_train)
eval_model(y_pts_test, enet_pts.predict(X_test_scaled), "ElasticNet (PTS)")

enet_ast = ElasticNet(alpha=0.1, l1_ratio=0.5)
enet_ast.fit(X_train_scaled, y_ast_train)
eval_model(y_ast_test, enet_ast.predict(X_test_scaled), "ElasticNet (AST)")

enet_reb = ElasticNet(alpha=0.1, l1_ratio=0.5)
enet_reb.fit(X_train_scaled, y_reb_train)
eval_model(y_reb_test, enet_reb.predict(X_test_scaled), "ElasticNet (REB)")


# =========================
# RANDOM FOREST (SAMPLED)
# =========================
sample_size = 100000
sample_idx = X_train.sample(n=sample_size, random_state=42).index

X_train_rf = X_train.loc[sample_idx]

rf_params = dict(
    n_estimators=60,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

# PTS
rf_pts = RandomForestRegressor(**rf_params)
rf_pts.fit(X_train_rf, y_pts_train.loc[sample_idx])
eval_model(y_pts_test, rf_pts.predict(X_test), "RF (PTS)")

# AST
rf_ast = RandomForestRegressor(**rf_params)
rf_ast.fit(X_train_rf, y_ast_train.loc[sample_idx])
eval_model(y_ast_test, rf_ast.predict(X_test), "RF (AST)")

# REB
rf_reb = RandomForestRegressor(**rf_params)
rf_reb.fit(X_train_rf, y_reb_train.loc[sample_idx])
eval_model(y_reb_test, rf_reb.predict(X_test), "RF (REB)")


# =========================
# FEATURE IMPORTANCE
# =========================
importances = pd.Series(rf_pts.feature_importances_, index=X_train.columns)
print("\nTop 20 PTS Features:")
print(importances.sort_values(ascending=False).head(20))


# =========================
# SAVE MODELS
# =========================
joblib.dump(ridge_pts, "models/player_pts_model.pkl")
joblib.dump(ridge_ast, "models/player_ast_model.pkl")
joblib.dump(ridge_reb, "models/player_reb_model.pkl")

joblib.dump(imputer, "models/player_stats_imputer.pkl")
joblib.dump(scaler, "models/player_stats_scaler.pkl")

print("\nSaved models.")