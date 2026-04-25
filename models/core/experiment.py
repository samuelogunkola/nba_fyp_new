import pandas as pd
import joblib

from models.core.preprocess import clean_features, Preprocessor, drop_all_nan_train_columns
from models.core.split import time_split, split_X_y, multi_target_split
from models.core.metrics import (
    classification_metrics,
    regression_metrics,
    print_classification_results,
    print_regression_results,
)


# =========================
# CLASSIFICATION EXPERIMENT
# =========================
def run_classification_experiment(
    df,
    target_col,
    model,
    model_name,
    drop_cols=None,
    split_date="2019-01-01",
    date_col="date",
    scale=False,
    remove_win_leakage=True,
    save_model_path=None,
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

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    y = df[target_col].copy()

    X = clean_features(
        df,
        drop_cols=drop_cols,
        remove_win_leakage=remove_win_leakage
    )

    train_idx, test_idx = time_split(df, date_col=date_col, split_date=split_date)
    X_train, X_test, y_train, y_test = split_X_y(X, y, train_idx, test_idx)

    X_train, X_test, dropped_nan_cols = drop_all_nan_train_columns(X_train, X_test)

    preprocessor = Preprocessor(scale=scale)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    model.fit(X_train_proc, y_train)

    y_pred = model.predict(X_test_proc)

    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test_proc)[:, 1]
        except Exception:
            y_proba = None

    metrics = classification_metrics(y_test, y_pred, y_proba)
    print_classification_results(model_name, metrics)

    results = {
        "model_name": model_name,
        "task": "classification",
        "target": target_col,
        "split_date": split_date,
        "train_rows": len(y_train),
        "test_rows": len(y_test),
        "n_features": X_train.shape[1],
        "dropped_all_nan_cols": dropped_nan_cols,
        **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
    }

    artifacts = {
        "model": model,
        "preprocessor": preprocessor,
        "feature_columns": X_train.columns.tolist(),
        "results": results,
    }

    if save_model_path:
        joblib.dump(artifacts, save_model_path)
        print(f"\nSaved experiment artifacts to: {save_model_path}")

    return artifacts


# =========================
# REGRESSION EXPERIMENT
# =========================
def run_regression_experiment(
    df,
    target_col,
    model,
    model_name,
    drop_cols=None,
    split_date="2019-01-01",
    date_col="date",
    scale=False,
    remove_win_leakage=False,
    save_model_path=None,
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

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    y = df[target_col].copy()

    X = clean_features(
        df,
        drop_cols=drop_cols,
        remove_win_leakage=remove_win_leakage
    )

    train_idx, test_idx = time_split(df, date_col=date_col, split_date=split_date)
    X_train, X_test, y_train, y_test = split_X_y(X, y, train_idx, test_idx)

    X_train, X_test, dropped_nan_cols = drop_all_nan_train_columns(X_train, X_test)

    preprocessor = Preprocessor(scale=scale)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    model.fit(X_train_proc, y_train)

    y_pred = model.predict(X_test_proc)

    metrics = regression_metrics(y_test, y_pred)
    print_regression_results(model_name, metrics)

    results = {
        "model_name": model_name,
        "task": "regression",
        "target": target_col,
        "split_date": split_date,
        "train_rows": len(y_train),
        "test_rows": len(y_test),
        "n_features": X_train.shape[1],
        "dropped_all_nan_cols": dropped_nan_cols,
        **metrics,
    }

    artifacts = {
        "model": model,
        "preprocessor": preprocessor,
        "feature_columns": X_train.columns.tolist(),
        "results": results,
    }

    if save_model_path:
        joblib.dump(artifacts, save_model_path)
        print(f"\nSaved experiment artifacts to: {save_model_path}")

    return artifacts


# =========================
# MULTI-TARGET REGRESSION
# =========================
def run_multi_target_regression_experiments(
    df,
    target_cols,
    models_dict,
    drop_cols=None,
    split_date="2019-01-01",
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

    results_rows = []

    for target_col in target_cols:
        for model_name, model in models_dict.items():
            print(f"\nRunning {model_name} for target: {target_col}")

            artifacts = run_regression_experiment(
                df=df,
                target_col=target_col,
                model=model,
                model_name=f"{model_name} ({target_col})",
                drop_cols=drop_cols,
                split_date=split_date,
                date_col=date_col,
                scale=scale,
                remove_win_leakage=remove_win_leakage,
                save_model_path=None,
            )

            results_rows.append(artifacts["results"])

    return pd.DataFrame(results_rows)