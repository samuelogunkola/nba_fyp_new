# pandas is used to load and work with the score matchup dataset
import pandas as pd

# numpy gives us maths tools — used here for RMSE and handling infinity values
import numpy as np

# joblib saves trained model objects to disk so they can be reloaded later without retraining
import joblib

# SimpleImputer fills missing values using the median of each training column
from sklearn.impute import SimpleImputer

# StandardScaler rescales features to zero mean and unit variance — important for Ridge
from sklearn.preprocessing import StandardScaler

# Ridge: linear regression with L2 regularisation — fast and reliable baseline for this problem
from sklearn.linear_model import Ridge

# two tree-based ensemble models to compare against the linear baseline:
# RandomForestRegressor: builds many independent trees and averages their predictions
# GradientBoostingRegressor: builds trees sequentially, each one fixing the previous one's errors
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# standard regression evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error


# load the processed score matchup dataset
# each row represents one game with pregame rolling features for both teams
df = pd.read_parquet("data/processed/score_matchups.parquet")

print("Loaded score_matchups shape:", df.shape)

# parse the date column into datetime objects so we can sort and split by date
df["date"] = pd.to_datetime(df["date"])

# sort oldest to newest so the time-based train/test split is chronologically correct
df = df.sort_values("date").reset_index(drop=True)


# separate out the two things we want to predict
# we predict home and away scores independently — each gets its own model
y_home = df["home_pts"].copy()   # actual home team points
y_away = df["away_pts"].copy()   # actual away team points


# list of columns to drop before building the feature matrix
# home_pts and away_pts are the targets — including them would be pure data leakage
# gameid, date, home, away are identifiers that carry no predictive signal
drop_cols = [
    "home_pts",   # target — must not be a feature
    "away_pts",   # target — must not be a feature
    "gameid",     # unique game ID — just a label, not a signal
    "date",       # raw date — temporal patterns come from rolling features, not the date itself
    "home",       # team name string — not numeric
    "away",       # team name string — not numeric
]

# drop the listed columns, then keep only numeric ones
# errors="ignore" means it won't crash if a listed column isn't present
X = df.drop(columns=drop_cols, errors="ignore").copy()
X = X.select_dtypes(include=[np.number]).copy()

# replace positive and negative infinity with NaN so the imputer can handle them
# infinity can appear in rolling division calculations (e.g. divide by zero)
X = X.replace([np.inf, -np.inf], np.nan)

# drop any column whose name contains "win" — these reveal game outcomes
# and would only be known after the final buzzer, not before tip-off
leakage_cols = [c for c in X.columns if "win" in c.lower()]
X = X.drop(columns=leakage_cols, errors="ignore")

print("Feature matrix shape:", X.shape)
print("Missing values:", int(X.isna().sum().sum()))


# split the data chronologically into training and test sets
# everything before 2019 is used for training, 2019 onwards is held out for testing
# a date-based split is essential here — random splits would let the model see "future" games
# during training which is unrealistic and makes results look artificially better
split_date = "2019-01-01"

train_idx = df["date"] < split_date    # True for pre-2019 rows
test_idx = df["date"] >= split_date    # True for 2019+ rows

# slice the feature matrix into train and test halves
X_train = X.loc[train_idx].copy()
X_test = X.loc[test_idx].copy()

# slice each target series into train and test halves
y_home_train = y_home.loc[train_idx]
y_home_test = y_home.loc[test_idx]

y_away_train = y_away.loc[train_idx]
y_away_test = y_away.loc[test_idx]

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)


# fit the imputer on training data only — it learns the median of each column from training
# then we apply those same learned medians to fill NaNs in the test set
# never fit on test data — that would leak test-set statistics into preprocessing
imputer = SimpleImputer(strategy="median")

# fit_transform learns medians and fills NaNs in one step, returning a numpy array
# we wrap the result back into a dataframe so column names are preserved for later use
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)

# transform applies the already-learned medians to the test set without refitting
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)


# fit the scaler on training data only — learns the mean and standard deviation per column
# StandardScaler makes every feature have zero mean and unit variance
# this is critical for Ridge which is sensitive to feature scale
# tree-based models (Random Forest, Gradient Boosting) don't need scaling, but we prepare it anyway
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)   # learn from train and apply
X_test_scaled = scaler.transform(X_test)          # apply the same learned stats to test


def evaluate(y_true, y_pred, name):
    # compute and print two regression metrics for a given set of predictions
    mae = mean_absolute_error(y_true, y_pred)

    # RMSE squares each error before averaging so large misses are penalised more than small ones
    # taking the square root brings the units back to points
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n{name}")
    print("MAE:", mae)    # average absolute error in points
    print("RMSE:", rmse)  # root mean squared error in points


# model 1: Ridge regression — fast linear baseline using scaled features
# Ridge adds L2 regularisation to prevent overfitting when features are correlated
# we train two separate models: one for home points, one for away points

ridge_home = Ridge()
ridge_home.fit(X_train_scaled, y_home_train)

pred_home_ridge = ridge_home.predict(X_test_scaled)
evaluate(y_home_test, pred_home_ridge, "Ridge (Home Points)")

ridge_away = Ridge()
ridge_away.fit(X_train_scaled, y_away_train)

pred_away_ridge = ridge_away.predict(X_test_scaled)
evaluate(y_away_test, pred_away_ridge, "Ridge (Away Points)")


# model 2: Random Forest — ensemble of 300 decision trees on unscaled features
# tree-based models split on thresholds rather than distances so they don't need scaling
# n_estimators=300 means 300 trees — more trees = more stable predictions but slower to train
# max_depth=12 limits how deep each tree grows to prevent memorising the training set
# n_jobs=-1 uses all CPU cores to parallelise tree building and speed up training

rf_home = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

# note: X_train (unscaled) is used here, not X_train_scaled
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


# model 3: Gradient Boosting — builds trees sequentially rather than in parallel
# each new tree focuses on correcting the mistakes of the previous trees
# this often gives better accuracy than Random Forest but takes longer to train
# we use the default hyperparameters here as an initial benchmark

gb_home = GradientBoostingRegressor(random_state=42)
gb_home.fit(X_train, y_home_train)

pred_home_gb = gb_home.predict(X_test)
evaluate(y_home_test, pred_home_gb, "Gradient Boosting (Home Points)")

gb_away = GradientBoostingRegressor(random_state=42)
gb_away.fit(X_train, y_away_train)

pred_away_gb = gb_away.predict(X_test)
evaluate(y_away_test, pred_away_gb, "Gradient Boosting (Away Points)")


# save the Random Forest models to disk — chosen as the main saved models for later use
# joblib serialises the fitted model objects to .pkl files that can be reloaded instantly
joblib.dump(rf_home, "models/home_score_model.pkl")
joblib.dump(rf_away, "models/away_score_model.pkl")

print("\nSaved score models.")
