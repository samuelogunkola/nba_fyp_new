# pandas is used to load and work with the player stats dataset
import pandas as pd

# numpy gives us maths tools — used here for RMSE calculation and handling infinity values
import numpy as np

# joblib saves trained model objects to disk so they can be reloaded later without retraining
import joblib

# SimpleImputer fills missing values using the median of each column
from sklearn.impute import SimpleImputer

# StandardScaler rescales features to zero mean and unit variance — needed for Ridge and ElasticNet
from sklearn.preprocessing import StandardScaler

# the regression models being compared in this script
# Ridge: linear regression with L2 regularisation — penalises large coefficients
# ElasticNet: combines both L1 (Lasso) and L2 (Ridge) regularisation
from sklearn.linear_model import Ridge, ElasticNet

# RandomForestRegressor: an ensemble of many decision trees that vote on the final prediction
from sklearn.ensemble import RandomForestRegressor

# standard regression evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error


# load the full processed player stats dataset from a parquet file
# parquet loads much faster than CSV for large files
df = pd.read_parquet("data/processed/player_stats_dataset.parquet")

print("Loaded player_stats_dataset shape:", df.shape)

# parse the date column into datetime objects so we can sort and split by date
df["date"] = pd.to_datetime(df["date"])

# sort oldest to newest — critical so the time-based split always puts past games in train
# and future games in test, never the other way around
df = df.sort_values("date").reset_index(drop=True)


# separate out the three target columns we want to predict
# each one gets its own series that we'll split into train/test below
y_pts = df["pts"]   # points scored
y_ast = df["ast"]   # assists
y_reb = df["reb"]   # rebounds


# build the feature matrix by dropping everything that shouldn't be a model input
# this includes the targets themselves (leakage) and non-informative identifier columns
drop_cols = ["pts", "ast", "reb", "gameid", "date", "player", "home", "away"]

X = df.drop(columns=drop_cols, errors="ignore")

# keep only numeric columns — models can't work with raw strings or object types
X = X.select_dtypes(include=[np.number])

# replace positive and negative infinity with NaN
# infinities can appear in rolling division calculations and will break the imputer
X = X.replace([np.inf, -np.inf], np.nan)

# drop any column whose name contains "win" — these reveal whether the team won the game
# which is only known after the game ends, making them classic data leakage
leakage_cols = [c for c in X.columns if "win" in c.lower()]
X = X.drop(columns=leakage_cols, errors="ignore")

print("Feature matrix shape:", X.shape)


# split the data into training and test sets using a date cutoff
# everything before 2019 goes into training, everything from 2019 onwards is held out for testing
# this mirrors real-world conditions — you always predict future games from past data
split_date = "2019-01-01"

train_idx = df["date"] < split_date    # boolean mask: True for pre-2019 rows
test_idx = df["date"] >= split_date    # boolean mask: True for 2019+ rows

# slice the feature matrix using the masks
X_train = X.loc[train_idx]
X_test = X.loc[test_idx]

# slice each target series into train and test halves
y_pts_train = y_pts.loc[train_idx]
y_pts_test = y_pts.loc[test_idx]

y_ast_train = y_ast.loc[train_idx]
y_ast_test = y_ast.loc[test_idx]

y_reb_train = y_reb.loc[train_idx]
y_reb_test = y_reb.loc[test_idx]

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)


# fit the imputer on training data only — it learns the median of each column from training
# then we apply those same medians to the test set
# never fit on test data — that would let test-set statistics influence preprocessing (leakage)
imputer = SimpleImputer(strategy="median")

# fit_transform learns medians and fills NaNs in one step, then wraps the result back into a dataframe
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)

# transform applies the already-learned medians to the test set without refitting
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)


# fit the scaler on training data only — it learns the mean and standard deviation per column
# StandardScaler makes every feature have zero mean and unit variance
# this is important for Ridge and ElasticNet because they're sensitive to feature scale
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)   # learn stats from train and apply
X_test_scaled = scaler.transform(X_test)          # apply the same learned stats to test


def eval_model(y_true, y_pred, name):
    # compute and print MAE and RMSE for a given set of predictions
    mae = mean_absolute_error(y_true, y_pred)

    # RMSE squares each error before averaging — large misses are penalised more than small ones
    # we take the square root at the end to bring the units back to the original stat (e.g. points)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n{name}")
    print("MAE:", mae)
    print("RMSE:", rmse)


# train and evaluate a separate Ridge model for each of the three targets
# Ridge adds a penalty to large coefficients which prevents overfitting — good for this problem
# because many rolling features are correlated with each other

# points model
ridge_pts = Ridge()
ridge_pts.fit(X_train_scaled, y_pts_train)
eval_model(y_pts_test, ridge_pts.predict(X_test_scaled), "Ridge (PTS)")

# assists model
ridge_ast = Ridge()
ridge_ast.fit(X_train_scaled, y_ast_train)
eval_model(y_ast_test, ridge_ast.predict(X_test_scaled), "Ridge (AST)")

# rebounds model
ridge_reb = Ridge()
ridge_reb.fit(X_train_scaled, y_reb_train)
eval_model(y_reb_test, ridge_reb.predict(X_test_scaled), "Ridge (REB)")


# train and evaluate ElasticNet models as an alternative to Ridge
# ElasticNet blends L1 (which can zero out features entirely) and L2 (Ridge-style) regularisation
# alpha=0.1 controls the overall regularisation strength — lower = less regularisation
# l1_ratio=0.5 means an even 50/50 blend of L1 and L2 penalties

# points model
enet_pts = ElasticNet(alpha=0.1, l1_ratio=0.5)
enet_pts.fit(X_train_scaled, y_pts_train)
eval_model(y_pts_test, enet_pts.predict(X_test_scaled), "ElasticNet (PTS)")

# assists model
enet_ast = ElasticNet(alpha=0.1, l1_ratio=0.5)
enet_ast.fit(X_train_scaled, y_ast_train)
eval_model(y_ast_test, enet_ast.predict(X_test_scaled), "ElasticNet (AST)")

# rebounds model
enet_reb = ElasticNet(alpha=0.1, l1_ratio=0.5)
enet_reb.fit(X_train_scaled, y_reb_train)
eval_model(y_reb_test, enet_reb.predict(X_test_scaled), "ElasticNet (REB)")


# Random Forest is more powerful but much slower to train on the full dataset
# so we randomly sample 100,000 rows from the training set to keep it manageable
# this is large enough to give reliable results while avoiding hour-long training times
sample_size = 100000
sample_idx = X_train.sample(n=sample_size, random_state=42).index

# slice both the features and targets down to the sampled rows only
X_train_rf = X_train.loc[sample_idx]

# shared hyperparameters for all three Random Forest models
# using a dict and ** unpacking keeps the code DRY (don't repeat yourself)
rf_params = dict(
    n_estimators=60,         # number of trees to build — more trees = more stable but slower
    max_depth=10,            # maximum depth of each tree — limits overfitting
    min_samples_split=20,    # a node must have at least 20 samples before it can split further
    min_samples_leaf=10,     # each leaf must contain at least 10 samples — prevents tiny noisy leaves
    random_state=42,         # fixed seed for reproducibility
    n_jobs=-1                # use all available CPU cores to speed up training
)

# points model — note we use X_test (unscaled) for Random Forest
# tree-based models don't need scaling because they split on thresholds, not distances
rf_pts = RandomForestRegressor(**rf_params)
rf_pts.fit(X_train_rf, y_pts_train.loc[sample_idx])
eval_model(y_pts_test, rf_pts.predict(X_test), "RF (PTS)")

# assists model
rf_ast = RandomForestRegressor(**rf_params)
rf_ast.fit(X_train_rf, y_ast_train.loc[sample_idx])
eval_model(y_ast_test, rf_ast.predict(X_test), "RF (AST)")

# rebounds model
rf_reb = RandomForestRegressor(**rf_params)
rf_reb.fit(X_train_rf, y_reb_train.loc[sample_idx])
eval_model(y_reb_test, rf_reb.predict(X_test), "RF (REB)")


# feature importance tells us which input columns the Random Forest relied on most
# for predicting points — useful for understanding what drives scoring predictions
# higher importance = the model used that feature more often and at higher splits in the trees
importances = pd.Series(rf_pts.feature_importances_, index=X_train.columns)
print("\nTop 20 PTS Features:")
print(importances.sort_values(ascending=False).head(20))


# save the Ridge models to disk — these are the ones used in the deployed dashboard
# joblib.dump serialises the model object to a .pkl file
joblib.dump(ridge_pts, "models/player_pts_model.pkl")
joblib.dump(ridge_ast, "models/player_ast_model.pkl")
joblib.dump(ridge_reb, "models/player_reb_model.pkl")

# also save the imputer and scaler so inference can apply the exact same transformations
# at prediction time — without these the feature values would be on the wrong scale
joblib.dump(imputer, "models/player_stats_imputer.pkl")
joblib.dump(scaler, "models/player_stats_scaler.pkl")

print("\nSaved models.")
