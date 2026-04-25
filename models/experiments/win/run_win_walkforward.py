import pandas as pd
from sklearn.linear_model import LogisticRegression

from models.core.preprocess import clean_features, Preprocessor, drop_all_nan_train_columns
from models.core.split import rolling_window_splits
from models.core.metrics import classification_metrics


# =========================
# LOAD DATA
# =========================
df = pd.read_parquet("data/processed/win_matchups.parquet")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

print("Loaded data:", df.shape)


# =========================
# SETUP
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


# =========================
# GENERATE SPLITS
# =========================
splits = rolling_window_splits(df, season_col="season", min_train_seasons=3)

print("Number of splits:", len(splits))


# =========================
# RUN WALK-FORWARD
# =========================
results = []

for i, split in enumerate(splits):
    print(f"\n=== Split {i+1} ===")
    print("Train seasons:", split["train_seasons"])
    print("Test season:", split["test_season"])

    train_idx = split["train_idx"]
    test_idx = split["test_idx"]

    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    # drop NaN cols
    X_train, X_test, _ = drop_all_nan_train_columns(X_train, X_test)

    # preprocess
    preprocessor = Preprocessor(scale=True)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # model
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_train_proc, y_train)

    y_pred = model.predict(X_test_proc)
    y_proba = model.predict_proba(X_test_proc)[:, 1]

    metrics = classification_metrics(y_test, y_pred, y_proba)

    print("Accuracy:", round(metrics["accuracy"], 4))
    print("F1:", round(metrics["f1"], 4))
    print("ROC-AUC:", round(metrics["roc_auc"], 4))

    results.append({
        "test_season": split["test_season"],
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "log_loss": metrics["log_loss"],
    })


# =========================
# RESULTS TABLE
# =========================
results_df = pd.DataFrame(results)

print("\n=== Walk-Forward Results ===")
print(results_df)

print("\nAverage Performance:")
print(results_df.mean())