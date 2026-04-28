# Path lets us build file paths in a clean, OS-independent way
from pathlib import Path

# joblib saves trained model objects to disk so they can be reloaded later without retraining
import joblib

# numpy is used for the correlation matrix upper-triangle calculation
import numpy as np

# pandas is used to load, sort, and manipulate the matchup dataset
import pandas as pd

# ColumnTransformer applies a preprocessing pipeline to a specific set of columns
from sklearn.compose import ColumnTransformer

# the three classifier families being compared in this experiment:
# - GradientBoostingClassifier: builds trees sequentially, each fixing the last one's errors
# - RandomForestClassifier: builds many independent trees and votes on the final answer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# SimpleImputer fills in missing values using the median of each training column
from sklearn.impute import SimpleImputer

# LogisticRegression: a linear model for binary classification — fast and interpretable baseline
from sklearn.linear_model import LogisticRegression

# all the classification metrics we need to evaluate model performance
from sklearn.metrics import (
    accuracy_score,        # fraction of predictions that were exactly correct
    confusion_matrix,      # 2x2 table of true/false positives and negatives
    f1_score,              # harmonic mean of precision and recall
    precision_score,       # of all predicted home wins, how many were actually home wins?
    recall_score,          # of all actual home wins, how many did the model catch?
    roc_auc_score,         # how well the model separates wins from losses at all thresholds
    classification_report, # formatted table of all per-class metrics in one block
)

# Pipeline chains preprocessing and modelling steps into a single object
from sklearn.pipeline import Pipeline

# StandardScaler rescales each feature to zero mean and unit variance — needed for Logistic Regression
from sklearn.preprocessing import StandardScaler

# VarianceThreshold drops features that barely vary across rows — constant columns teach nothing
from sklearn.feature_selection import VarianceThreshold


# path to the processed win matchup dataset
DATA_PATH = Path("data/processed/win_matchups.parquet")

# folder where all trained models and result CSVs will be saved
ARTIFACT_DIR = Path("models/experiments/win/artifacts")

# create the folder if it doesn't already exist
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# the column we're trying to predict: 1 = home team won, 0 = away team won
TARGET = "home_win"

# columns we always drop — these are identifiers or the target itself, not usable features
DROP_COLS = ["gameid", "date", "season", TARGET]


def load_data(path):
    # support both parquet and CSV — check the file extension to decide which reader to use
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    print(f"Loaded win_matchups: {df.shape}")

    if "date" in df.columns:
        # parse dates so we can sort the dataset chronologically
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        # sort oldest first — essential so the 80/20 time split is chronologically correct
        df = df.sort_values("date").reset_index(drop=True)

    return df


def is_leakage_col(col):
    # returns True if this column looks like it contains post-game information
    # that would only be known after the final whistle — never before tip-off

    # rolling averages and exponential moving averages are safe — they only look backwards
    # columns starting with "diff_" are pre-game matchup differences, also safe
    allowed_if_rolling = "_roll_" in col or "_exp_" in col or col.startswith("diff_")

    if allowed_if_rolling:
        return False

    # these strings in a column name strongly suggest it reveals the game's outcome
    # e.g. "home_win" is literally the label we're predicting — obvious leakage
    # "plus_minus" is only available after the game ends
    leakage_terms = [
        "home_win",
        "away_win",
        "winner",
        "result",
        "home_pts",
        "away_pts",
        "point_spread",
        "total_points",
        "plus_minus",
    ]

    # always block the exact target column
    if col == TARGET:
        return True

    # block if any of the leakage terms appear anywhere in the column name
    return any(term in col for term in leakage_terms)


def get_feature_columns(df):
    # build the list of columns that are safe to use as model inputs
    features = []  # columns we'll use
    blocked = []   # columns we removed due to suspected leakage

    for col in df.columns:
        # skip identifiers and the target column
        if col in DROP_COLS:
            continue

        # skip non-numeric columns — classifiers can't work with raw strings
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        # run the leakage check — move anything suspicious to the blocked list
        if is_leakage_col(col):
            blocked.append(col)
            continue

        features.append(col)

    # print the blocked list so we have a clear audit trail of what was removed and why
    if blocked:
        print("\nBlocked possible leakage columns:")
        for col in blocked:
            print("-", col)

    print(f"\nUsing {len(features)} numeric pre-game features")
    return features


def make_time_split(df, test_size=0.2):
    # split chronologically: first 80% of rows for training, last 20% for testing
    # because data is sorted by date, this always means training on older games
    # and testing on newer ones — just like real-world prediction conditions
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    return train_df, test_df


def drop_high_corr_columns(X_train, X_test, threshold=0.95):
    # remove features that are extremely highly correlated with each other
    # two features that are 95%+ correlated carry almost identical information
    # keeping both wastes computation and can cause instability in linear models

    # compute the absolute pairwise correlation between every feature in the training set
    corr = X_train.corr().abs()

    # np.triu extracts just the upper triangle of the matrix so we don't double-count pairs
    # k=1 means we exclude the diagonal (a feature is always 100% correlated with itself)
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    # find any column that has a correlation above the threshold with at least one other column
    to_drop = [
        col for col in upper.columns
        if any(upper[col] > threshold)
    ]

    # drop those columns from both sets so their shapes always stay consistent
    X_train = X_train.drop(columns=to_drop, errors="ignore")
    X_test = X_test.drop(columns=to_drop, errors="ignore")

    print(f"Dropped {len(to_drop)} highly correlated features")
    print(f"Remaining features: {X_train.shape[1]}")

    return X_train, X_test


def build_preprocessor(feature_cols):
    # build a preprocessing pipeline that runs before every model
    # chaining steps in a Pipeline guarantees they always run in the same order
    numeric_pipeline = Pipeline(
        steps=[
            # fill NaN values with the median of each column (learned from training data only)
            ("imputer", SimpleImputer(strategy="median")),
            # remove any columns with zero variance — completely constant columns are useless
            ("variance", VarianceThreshold()),
            # rescale to zero mean and unit variance — critical for Logistic Regression
            ("scaler", StandardScaler()),
        ]
    )

    # ColumnTransformer applies the pipeline only to the specified feature columns
    # remainder="drop" discards any other columns rather than passing them through unchanged
    return ColumnTransformer(
        transformers=[("num", numeric_pipeline, feature_cols)],
        remainder="drop",
    )


def evaluate_classifier(model_name, y_true, y_pred, y_prob):
    # compute all classification metrics and return them as a dictionary
    result = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        # zero_division=0 means we get 0.0 instead of a warning when a class is never predicted
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        # ROC-AUC requires probability scores, not just hard 0/1 predictions
        "roc_auc": roc_auc_score(y_true, y_prob),
    }

    # print a clean per-model summary to the console while the experiment is running
    print(f"\n=== {model_name} ===")
    print(f"Accuracy:  {result['accuracy']:.4f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall:    {result['recall']:.4f}")
    print(f"F1:        {result['f1']:.4f}")
    print(f"ROC-AUC:   {result['roc_auc']:.4f}")

    # confusion matrix: rows = actual, columns = predicted
    # top-left = correct away wins, bottom-right = correct home wins
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # classification report gives per-class precision, recall, and F1 in a readable table
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    return result


def get_models(feature_cols):
    # build all three model pipelines and return them in a dictionary
    # each entry maps a readable model name to a fitted-ready Pipeline object
    return {
        # Logistic Regression: fast linear baseline — good for understanding what's learnable
        # max_iter=5000 gives the solver enough iterations to converge on this dataset size
        # class_weight="balanced" adjusts for any class imbalance between home and away wins
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(feature_cols)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),

        # Random Forest: builds many independent decision trees and combines their votes
        # n_estimators=250 means 250 trees — more trees = more stable but slower to train
        # max_depth=10 limits tree depth to prevent memorising the training set
        # min_samples_leaf=5 means a leaf node needs at least 5 games to form — prevents tiny splits
        # n_jobs=-1 uses all available CPU cores to speed up training
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(feature_cols)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        max_depth=10,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        # Gradient Boosting: builds trees sequentially, each one correcting previous errors
        # learning_rate=0.05 means each tree makes small corrections — more stable than a high rate
        # max_depth=2 keeps each individual tree very shallow to avoid overfitting
        "Gradient Boosting": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(feature_cols)),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=150,
                        learning_rate=0.05,
                        max_depth=2,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def save_model(model, filename):
    # save a trained model pipeline to disk using joblib
    # the saved file can be reloaded later with joblib.load() for inference
    path = ARTIFACT_DIR / filename
    joblib.dump(model, path)
    print(f"Saved model: {path}")


def main():
    # load and sort the full win matchup dataset
    df = load_data(DATA_PATH)

    # make sure the target column is actually in the data before doing any work
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found. Available columns include: {df.columns[:30].tolist()}")

    # split into training (80%) and test (20%) sets chronologically
    train_df, test_df = make_time_split(df)

    # identify the safe, leakage-free feature columns
    feature_cols = get_feature_columns(df)

    # extract the feature matrices and replace any infinity values with NaN
    # (infinities can appear in rolling calculations and will break the imputer)
    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # extract the target labels and cast to int (0 or 1)
    y_train = train_df[TARGET].astype(int)
    y_test = test_df[TARGET].astype(int)

    # remove highly correlated features to reduce redundancy
    X_train, X_test = drop_high_corr_columns(X_train, X_test)

    # update the feature column list to reflect what survived the correlation filter
    feature_cols = list(X_train.columns)

    # build all three model pipelines using the final feature set
    models = get_models(feature_cols)

    # list to collect one results dictionary per model
    results = []

    # keep a copy of the test rows with identifiers so we can attach predictions alongside them
    predictions = test_df[[c for c in ["gameid", "date", "season", TARGET] if c in test_df.columns]].copy()

    # track which model had the best F1 score so we can report the winner at the end
    best_model_name = None
    best_f1 = -1

    # train and evaluate each model in turn
    for model_name, model in models.items():
        # train on the processed training set
        model.fit(X_train, y_train)

        # generate hard class predictions (0 or 1) for the test set
        y_pred = model.predict(X_test)

        # get probability estimates if the model supports them
        # predict_proba returns probabilities for both classes — [:, 1] gives the home win probability
        # this is needed for ROC-AUC which requires probabilities rather than hard predictions
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            # fallback for models that don't support predict_proba (shouldn't happen here)
            y_prob = y_pred

        # evaluate the model and print results to the console
        result = evaluate_classifier(model_name, y_test, y_pred, y_prob)
        results.append(result)

        # save the trained model pipeline to disk with a filename based on the model name
        safe_name = model_name.lower().replace(" ", "_")
        save_model(model, f"{safe_name}_home_win.pkl")

        # attach this model's predictions and probabilities to the predictions dataframe
        predictions[f"{safe_name}_pred"] = y_pred
        predictions[f"{safe_name}_prob"] = y_prob

        # keep track of the best model by F1 score
        if result["f1"] > best_f1:
            best_f1 = result["f1"]
            best_model_name = model_name

    # combine all results dictionaries into a single dataframe — one row per model
    results_df = pd.DataFrame(results)
    results_path = ARTIFACT_DIR / "win_results.csv"
    predictions_path = ARTIFACT_DIR / "win_prediction_details.csv"

    # save both CSVs to the artifact directory
    results_df.to_csv(results_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    # print the final summary table to the console
    print("\n" + "=" * 60)
    print("WIN MODEL RESULTS SUMMARY")
    print("=" * 60)
    print(results_df)

    # announce the winning model
    print(f"\nBest model by F1: {best_model_name} ({best_f1:.4f})")

    # confirm where the output files were written
    print("\nSaved:")
    print(f"- {results_path}")
    print(f"- {predictions_path}")


# only run main() when this script is executed directly from the command line
# if it's imported by another script, main() won't run automatically
if __name__ == "__main__":
    main()
