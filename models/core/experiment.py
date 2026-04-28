# pandas is used at the end to collect all results into a tidy dataframe
import pandas as pd

# joblib is used to save trained model artifacts to disk as .pkl files
# this means we don't have to retrain the model every time we want to use it
import joblib

# these imports pull in our own helper modules:
# - preprocess: cleans feature columns and applies scaling/imputation
# - split: handles dividing the data into train and test sets by date
# - metrics: calculates and prints evaluation scores like F1, MAE, R²
from models.core.preprocess import clean_features, Preprocessor, drop_all_nan_train_columns
from models.core.split import time_split, split_X_y, multi_target_split
from models.core.metrics import (
    classification_metrics,
    regression_metrics,
    print_classification_results,
    print_regression_results,
)


# this function runs a complete classification experiment from raw dataframe to saved results
# classification means we're predicting a category (e.g. "home win" vs "away win")
# all the messy steps — cleaning, splitting, preprocessing, fitting, evaluating — happen here
def run_classification_experiment(
    df,                          # the full dataset as a pandas dataframe
    target_col,                  # the name of the column we want to predict (e.g. "home_win")
    model,                       # a scikit-learn model object (e.g. RandomForestClassifier())
    model_name,                  # a plain-English name used in printed output and results
    drop_cols=None,              # any extra columns to remove before training
    split_date="2019-01-01",    # games before this date go into train, games after go into test
    date_col="date",             # the name of the date column in the dataframe
    scale=False,                 # whether to standardise features to zero mean and unit variance
    remove_win_leakage=True,     # strip out same-game win/loss columns to prevent data leakage
    save_model_path=None,        # if provided, the trained model is saved to this file path
):
    """
    Runs a full classification experiment:
    - clean features
    - time split
    - drop train-only NaN columns
    - preprocess
    - fit model
    - predict
    - evaluate
    """

    # make sure the target column actually exists before doing any work
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    # separate the target column (what we want to predict) from the rest of the data
    y = df[target_col].copy()

    # clean_features removes identifiers, leakage columns, and non-numeric columns
    # it returns just the feature matrix X that the model will actually learn from
    X = clean_features(
        df,
        drop_cols=drop_cols,
        remove_win_leakage=remove_win_leakage
    )

    # time_split returns the row indices for train and test based on the split date
    # using a date-based split rather than a random one prevents the model from
    # accidentally learning from "future" games when predicting "past" ones
    train_idx, test_idx = time_split(df, date_col=date_col, split_date=split_date)

    # use those indices to slice X and y into four separate arrays
    X_train, X_test, y_train, y_test = split_X_y(X, y, train_idx, test_idx)

    # remove any columns that are entirely NaN in the training set
    # if a column has no data at all in training it can't teach the model anything
    # we also drop those same columns from the test set to keep the shapes consistent
    X_train, X_test, dropped_nan_cols = drop_all_nan_train_columns(X_train, X_test)

    # the Preprocessor handles imputing missing values (filling NaNs) and optionally scaling
    # fit_transform learns the imputation values from the training set and applies them
    preprocessor = Preprocessor(scale=scale)
    X_train_proc = preprocessor.fit_transform(X_train)

    # transform applies the same learned imputation to the test set
    # we never fit on the test set — that would be leakage
    X_test_proc = preprocessor.transform(X_test)

    # train the model on the processed training data
    model.fit(X_train_proc, y_train)

    # generate class predictions (0 or 1) for every row in the test set
    y_pred = model.predict(X_test_proc)

    # also try to get probability estimates if the model supports them
    # predict_proba returns a probability for each class; [:, 1] gives us the positive class probability
    # this is needed to calculate ROC-AUC, which requires probabilities not just hard predictions
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test_proc)[:, 1]
        except Exception:
            # some models claim to have predict_proba but fail — catch that gracefully
            y_proba = None

    # compute accuracy, precision, recall, F1, ROC-AUC, and the confusion matrix
    metrics = classification_metrics(y_test, y_pred, y_proba)

    # print a formatted summary of the results to the console
    print_classification_results(model_name, metrics)

    # build a flat dictionary of everything we want to record about this run
    results = {
        "model_name": model_name,
        "task": "classification",
        "target": target_col,
        "split_date": split_date,
        "train_rows": len(y_train),       # how many games were used for training
        "test_rows": len(y_test),          # how many games were held out for testing
        "n_features": X_train.shape[1],   # how many input features the model used
        "dropped_all_nan_cols": dropped_nan_cols,
        # unpack all metrics except the confusion matrix (it's a numpy array, needs special handling)
        **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
        # convert the confusion matrix to a plain Python list so it can be serialised to JSON/CSV
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
    }

    # bundle everything needed to reproduce or use this model later into one dictionary
    artifacts = {
        "model": model,                                    # the trained scikit-learn model object
        "preprocessor": preprocessor,                      # the fitted preprocessor (for inference)
        "feature_columns": X_train.columns.tolist(),       # the exact feature names used in training
        "results": results,                                # the evaluation metrics dictionary
    }

    # if a save path was provided, persist the artifacts to disk with joblib
    # the saved file can be loaded later with joblib.load() for inference or inspection
    if save_model_path:
        joblib.dump(artifacts, save_model_path)
        print(f"\nSaved experiment artifacts to: {save_model_path}")

    return artifacts


# this function runs a complete regression experiment from raw dataframe to saved results
# regression means we're predicting a continuous number (e.g. points scored, not win/loss)
# the structure is almost identical to the classification version above
def run_regression_experiment(
    df,                          # the full dataset as a pandas dataframe
    target_col,                  # the name of the column we want to predict (e.g. "home_pts")
    model,                       # a scikit-learn model object (e.g. Ridge())
    model_name,                  # a plain-English name used in printed output and results
    drop_cols=None,              # any extra columns to remove before training
    split_date="2019-01-01",    # games before this date go into train, games after go into test
    date_col="date",             # the name of the date column in the dataframe
    scale=False,                 # whether to standardise features before fitting
    remove_win_leakage=False,    # for regression we typically don't need to strip win columns
    save_model_path=None,        # if provided, the trained model is saved to this file path
):
    """
    Runs a full regression experiment:
    - clean features
    - time split
    - drop train-only NaN columns
    - preprocess
    - fit model
    - predict
    - evaluate
    """

    # make sure the target column actually exists before doing any work
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    # separate the target column from the rest of the data
    y = df[target_col].copy()

    # clean the feature matrix — remove identifiers, leakage columns, and non-numeric columns
    X = clean_features(
        df,
        drop_cols=drop_cols,
        remove_win_leakage=remove_win_leakage
    )

    # split rows into train (before split_date) and test (after split_date) by index
    train_idx, test_idx = time_split(df, date_col=date_col, split_date=split_date)
    X_train, X_test, y_train, y_test = split_X_y(X, y, train_idx, test_idx)

    # drop any columns that are completely empty in the training set
    X_train, X_test, dropped_nan_cols = drop_all_nan_train_columns(X_train, X_test)

    # fit the imputer (and optional scaler) on training data only, then apply to both sets
    preprocessor = Preprocessor(scale=scale)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # train the regression model
    model.fit(X_train_proc, y_train)

    # generate numeric predictions for each test row
    y_pred = model.predict(X_test_proc)

    # compute MAE, RMSE, and R² to measure how close predictions were to real values
    metrics = regression_metrics(y_test, y_pred)

    # print a formatted summary to the console
    print_regression_results(model_name, metrics)

    # record everything about this experiment run into a flat dictionary
    results = {
        "model_name": model_name,
        "task": "regression",
        "target": target_col,
        "split_date": split_date,
        "train_rows": len(y_train),
        "test_rows": len(y_test),
        "n_features": X_train.shape[1],
        "dropped_all_nan_cols": dropped_nan_cols,
        # unpack all regression metrics (MAE, RMSE, R²) directly into the results dict
        **metrics,
    }

    # bundle the trained model, preprocessor, feature names, and results for later use
    artifacts = {
        "model": model,
        "preprocessor": preprocessor,
        "feature_columns": X_train.columns.tolist(),
        "results": results,
    }

    # save everything to disk if a path was given
    if save_model_path:
        joblib.dump(artifacts, save_model_path)
        print(f"\nSaved experiment artifacts to: {save_model_path}")

    return artifacts


# this function runs multiple regression experiments in a nested loop:
# for every target column (e.g. pts, reb, ast) and every model (e.g. Ridge, RandomForest),
# it calls run_regression_experiment and collects all the results into one dataframe
def run_multi_target_regression_experiments(
    df,                          # the full dataset
    target_cols,                 # list of columns to predict, e.g. ["pts", "reb", "ast"]
    models_dict,                 # dict mapping model names to model objects, e.g. {"Ridge": Ridge()}
    drop_cols=None,              # columns to drop from features across all experiments
    split_date="2019-01-01",    # train/test cutoff date applied to every experiment
    date_col="date",
    scale=False,
    remove_win_leakage=False,
):
    """
    Runs separate regression experiments for multiple targets.

    Example:
    target_cols = ["pts", "ast", "reb"]
    models_dict = {
        "Ridge": Ridge(),
        "RF": RandomForestRegressor(...)
    }
    """

    # we'll collect the results dict from each experiment and combine them at the end
    results_rows = []

    # outer loop: iterate over each stat we want to predict
    for target_col in target_cols:
        # inner loop: try every model for this target
        for model_name, model in models_dict.items():
            print(f"\nRunning {model_name} for target: {target_col}")

            # run a full regression experiment for this target/model combination
            artifacts = run_regression_experiment(
                df=df,
                target_col=target_col,
                model=model,
                # include the target name in the model label so results are easy to tell apart
                model_name=f"{model_name} ({target_col})",
                drop_cols=drop_cols,
                split_date=split_date,
                date_col=date_col,
                scale=scale,
                remove_win_leakage=remove_win_leakage,
                save_model_path=None,  # don't save individual models in the multi-target loop
            )

            # add this experiment's results row to our growing list
            results_rows.append(artifacts["results"])

    # convert the list of result dicts into a tidy dataframe — one row per experiment
    return pd.DataFrame(results_rows)
