# pandas is used to load and work with the win matchup dataset
import pandas as pd

# numpy gives us maths tools — used here for handling infinity values and RMSE calculation
import numpy as np

# joblib saves trained model objects to disk so they can be reloaded later without retraining
import joblib

# SimpleImputer fills missing values using the median of each training column
from sklearn.impute import SimpleImputer

# StandardScaler rescales features to zero mean and unit variance — needed for Logistic Regression
from sklearn.preprocessing import StandardScaler

# the three classifiers being compared in this script:
# LogisticRegression: fast linear baseline — good for understanding what patterns exist
from sklearn.linear_model import LogisticRegression

# RandomForestClassifier: builds many trees independently and votes on the final answer
# GradientBoostingClassifier: builds trees sequentially, each one correcting the last
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# classification metrics used to judge how well each model is performing
from sklearn.metrics import (
    accuracy_score,     # fraction of predictions that were exactly correct
    f1_score,           # harmonic mean of precision and recall
    confusion_matrix,   # 2x2 table of true/false positives and negatives
    log_loss,           # penalises confident wrong probability predictions — lower is better
    brier_score_loss,   # measures how well-calibrated the probability estimates are (0 = perfect)
)


# load the processed win matchup dataset — each row represents one game
# with pregame rolling features for both teams and the actual outcome
df = pd.read_parquet("data/processed/win_matchups.parquet")

print("Loaded win_matchups shape:", df.shape)

# parse the date column into proper datetime objects so we can sort and split by date
df["date"] = pd.to_datetime(df["date"])

# sort oldest to newest — critical so the time-based split always puts past games in training
# and future games in testing, never the other way around
df = df.sort_values("date").reset_index(drop=True)


# the label we're predicting: 1 = home team won, 0 = away team won
y = df["home_win"].copy()


# columns to remove before building the feature matrix
# home_win is the target — including it would be data leakage
# the others are identifiers that carry no predictive signal
drop_cols = ["home_win", "gameid", "date", "home", "away"]

# drop the listed columns, then keep only numeric ones
# errors="ignore" means it won't crash if a listed column doesn't exist
X = df.drop(columns=drop_cols, errors="ignore").copy()
X = X.select_dtypes(include=[np.number]).copy()

# replace any infinity values with NaN so the imputer can handle them
# infinities can appear in rolling division calculations (e.g. dividing by zero)
X = X.replace([np.inf, -np.inf], np.nan)

# drop any column whose name contains "win" — these reveal whether a team won the game
# which is information only available after the game ends, making them leakage
leakage_keywords = ["win"]
leakage_cols = [c for c in X.columns if any(k in c.lower() for k in leakage_keywords)]

print("Win-related columns dropped:", len(leakage_cols))
X = X.drop(columns=leakage_cols, errors="ignore")

print("Feature matrix shape:", X.shape)
print("Missing values before split:", int(X.isna().sum().sum()))


# split into training and test sets using a date cutoff
# everything before 2019 is used for training, 2019 onwards is held out for testing
# this mirrors real-world conditions — you always train on past games and test on future ones
# a random split would let the model see "future" games during training, inflating results artificially
split_date = "2019-01-01"

train_idx = df["date"] < split_date    # True for all pre-2019 rows
test_idx = df["date"] >= split_date    # True for all 2019+ rows

# slice both the feature matrix and the target using the masks
X_train = X.loc[train_idx].copy()
X_test = X.loc[test_idx].copy()
y_train = y.loc[train_idx].copy()
y_test = y.loc[test_idx].copy()

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)


# find any columns that are completely empty (all NaN) in the training set
# these columns have nothing to teach the model and some sklearn functions reject them outright
all_nan_train_cols = X_train.columns[X_train.isna().all()].tolist()

if all_nan_train_cols:
    # drop those columns from both sets so their shapes stay consistent
    X_train = X_train.drop(columns=all_nan_train_cols)
    X_test = X_test.drop(columns=all_nan_train_cols)

print("Dropped all-NaN columns:", len(all_nan_train_cols))


# fit the imputer on training data only — it learns the median of each column from training
# then we apply those same medians to fill NaNs in the test set
# never fit on test data — that would allow test statistics to influence preprocessing (leakage)
imputer = SimpleImputer(strategy="median")

# fit_transform learns and applies in one step — we preserve the original index for alignment
X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index,
)

# transform applies the already-learned medians to the test set
X_test_imputed = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns,
    index=X_test.index,
)


# fit the scaler on training data only — learns the mean and standard deviation per column
# StandardScaler makes every feature have zero mean and unit variance
# Logistic Regression is sensitive to feature scale so this is essential for that model
# tree-based models don't need scaling but we keep separate scaled/unscaled versions for clarity
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imputed)   # learn from train and apply
X_test_scaled = scaler.transform(X_test_imputed)          # apply the same stats to test


# model 1: Logistic Regression — a simple linear classifier used as a baseline
# max_iter=2000 gives the solver enough iterations to converge on this dataset size
# uses the scaled feature matrix because it's sensitive to feature magnitude
lr = LogisticRegression(max_iter=2000, random_state=42)
lr.fit(X_train_scaled, y_train)

# generate hard class predictions (0 or 1) and probability estimates
y_pred_lr = lr.predict(X_test_scaled)

# predict_proba returns probabilities for both classes — [:, 1] gives the home win probability
# this is needed for log loss and Brier score which require probabilities, not hard predictions
y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

print("\n=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("F1:", f1_score(y_test, y_pred_lr))
print("Log Loss:", log_loss(y_test, y_proba_lr))
print("Brier Score:", brier_score_loss(y_test, y_proba_lr))
print(confusion_matrix(y_test, y_pred_lr))


# model 2: Random Forest — ensemble of 400 independent decision trees
# uses the imputed (unscaled) matrix — tree-based models don't need feature scaling
# n_estimators=400: more trees = more stable but slower to train
# max_depth=14: limits how deep each tree can grow to prevent memorising training data
# min_samples_split=10: a node needs at least 10 games before it can be split further
# min_samples_leaf=4: each leaf must contain at least 4 games — prevents tiny overfit leaves
# class_weight="balanced": adjusts for any imbalance between home wins and away wins
# n_jobs=-1: use all CPU cores to parallelise tree building
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=14,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

rf.fit(X_train_imputed, y_train)

y_pred_rf = rf.predict(X_test_imputed)
y_proba_rf = rf.predict_proba(X_test_imputed)[:, 1]

print("\n=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("F1:", f1_score(y_test, y_pred_rf))
print("Log Loss:", log_loss(y_test, y_proba_rf))
print("Brier Score:", brier_score_loss(y_test, y_proba_rf))
print(confusion_matrix(y_test, y_pred_rf))


# model 3: Gradient Boosting — builds trees one at a time, each focused on what the previous got wrong
# often gives the best accuracy of the three models but takes longer to train than Random Forest
# using default hyperparameters here as an initial comparison point
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train_imputed, y_train)

y_pred_gb = gb.predict(X_test_imputed)
y_proba_gb = gb.predict_proba(X_test_imputed)[:, 1]

print("\n=== Gradient Boosting ===")
print("Accuracy:", accuracy_score(y_test, y_pred_gb))
print("F1:", f1_score(y_test, y_pred_gb))
print("Log Loss:", log_loss(y_test, y_proba_gb))
print("Brier Score:", brier_score_loss(y_test, y_proba_gb))
print(confusion_matrix(y_test, y_pred_gb))


# feature importance from the Random Forest tells us which columns it relied on most
# for predicting home wins — higher importance = used more often at higher tree splits
# this is useful for understanding which pregame stats actually matter
importances = pd.Series(rf.feature_importances_, index=X_train_imputed.columns)
top_features = importances.sort_values(ascending=False)

print("\nTop 20 Features:")
print(top_features.head(20))


# feature selection experiment: retrain Random Forest using only the top 50 features
# this tests whether a smaller, cleaner feature set performs as well as the full one
# fewer features can reduce overfitting and make the model faster to run
top_n = 50
selected_features = top_features.head(top_n).index

# slice the training and test matrices down to just the selected columns
X_train_sel = X_train_imputed[selected_features]
X_test_sel = X_test_imputed[selected_features]

# use the same hyperparameters as the full Random Forest for a fair comparison
rf_top50 = RandomForestClassifier(
    n_estimators=400,
    max_depth=14,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

rf_top50.fit(X_train_sel, y_train)

y_pred_sel = rf_top50.predict(X_test_sel)
y_proba_sel = rf_top50.predict_proba(X_test_sel)[:, 1]

print("\n=== Random Forest (Top 50 Features) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_sel))
print("F1:", f1_score(y_test, y_pred_sel))
print("Log Loss:", log_loss(y_test, y_proba_sel))
print("Brier Score:", brier_score_loss(y_test, y_proba_sel))
print(confusion_matrix(y_test, y_pred_sel))


# save the full Random Forest model along with the imputer and scaler
# these three files together are everything needed to make predictions on new data:
# 1. imputer: fills any missing values using the training-set medians
# 2. scaler: rescales features (only needed if running Logistic Regression for inference)
# 3. rf: the trained Random Forest classifier
joblib.dump(rf, "models/win_model.pkl")
joblib.dump(imputer, "models/win_imputer.pkl")
joblib.dump(scaler, "models/win_scaler.pkl")

print("\nSaved models.")
