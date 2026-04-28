# pandas is used to load and work with the matchup dataset
import pandas as pd

# LogisticRegression is the classifier used across every fold of the walk-forward evaluation
# it's a fast, interpretable linear model — a good choice for testing temporal consistency
from sklearn.linear_model import LogisticRegression

# our own helper modules for cleaning features, preprocessing, and splitting data
from models.core.preprocess import clean_features, Preprocessor, drop_all_nan_train_columns
from models.core.split import rolling_window_splits
from models.core.metrics import classification_metrics


# load the win matchup dataset from a parquet file (faster than CSV for large files)
df = pd.read_parquet("data/processed/win_matchups.parquet")

# parse the date column so we can sort rows chronologically
df["date"] = pd.to_datetime(df["date"])

# sort oldest to newest — the rolling window splits rely on this order being correct
df = df.sort_values("date").reset_index(drop=True)

print("Loaded data:", df.shape)


# the binary label we're trying to predict: 1 = home team won, 0 = away team won
target_col = "home_win"

# columns to strip out before building the feature matrix
# these are identifiers or the target itself — none of them should be model inputs
drop_cols = [
    "home_win",  # this IS the target — including it would be obvious leakage
    "gameid",    # a unique game identifier, carries no predictive signal
    "date",      # raw dates aren't useful features; temporal information comes from the rolling stats
    "home",      # team name string — not numeric, gets dropped by clean_features anyway
    "away",      # same as above
]

# clean_features strips out the drop columns, keeps only numeric columns,
# replaces infinity values with NaN, and optionally removes win-related leakage columns
X = clean_features(df, drop_cols=drop_cols)

# separate out the target column so we can use it as labels during training
y = df[target_col]


# generate all the walk-forward season splits
# this function expands the training window one season at a time:
#   split 1: train on seasons 1–3, test on season 4
#   split 2: train on seasons 1–4, test on season 5
#   ...and so on until we run out of seasons
# min_train_seasons=3 means we need at least 3 seasons before we start testing
splits = rolling_window_splits(df, season_col="season", min_train_seasons=3)

print("Number of splits:", len(splits))


# collect the results from every split into this list
# at the end we'll turn it into a dataframe to see how performance varied over time
results = []

# loop over every walk-forward split in order
for i, split in enumerate(splits):
    print(f"\n=== Split {i+1} ===")
    print("Train seasons:", split["train_seasons"])
    print("Test season:", split["test_season"])

    # retrieve the boolean row masks for this split
    train_idx = split["train_idx"]
    test_idx = split["test_idx"]

    # slice the feature matrix and labels using the masks
    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    # drop any columns that are completely empty in the training fold
    # these can't contribute anything to the model and some sklearn functions reject all-NaN columns
    X_train, X_test, _ = drop_all_nan_train_columns(X_train, X_test)

    # fit the imputer (and scaler) on the training fold only, then apply to both
    # scale=True means features are standardised to zero mean and unit variance
    # we never fit on test data — that would leak information about the test set into preprocessing
    preprocessor = Preprocessor(scale=True)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # train a fresh Logistic Regression model for this split
    # max_iter=2000 gives the solver enough iterations to converge on larger training folds
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_train_proc, y_train)

    # generate hard class predictions (0 or 1) on the held-out test season
    y_pred = model.predict(X_test_proc)

    # get probability estimates for the positive class (home win)
    # predict_proba returns probabilities for both classes — [:, 1] gives the home win probability
    # needed to compute ROC-AUC which requires probabilities, not hard predictions
    y_proba = model.predict_proba(X_test_proc)[:, 1]

    # compute accuracy, F1, ROC-AUC, and log loss for this fold
    metrics = classification_metrics(y_test, y_pred, y_proba)

    # print a quick summary for this split so we can monitor progress as the script runs
    print("Accuracy:", round(metrics["accuracy"], 4))
    print("F1:", round(metrics["f1"], 4))
    print("ROC-AUC:", round(metrics["roc_auc"], 4))

    # store the metrics for this fold — we use the test season as an identifier
    results.append({
        "test_season": split["test_season"],
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "log_loss": metrics["log_loss"],
    })


# convert all per-fold results into a dataframe — one row per season tested
results_df = pd.DataFrame(results)

# print the full table so we can see how the model performed year by year
print("\n=== Walk-Forward Results ===")
print(results_df)

# average across all folds to get a single summary of overall temporal performance
# this is the most honest estimate of how the model would perform in real-world use
print("\nAverage Performance:")
print(results_df.mean())
