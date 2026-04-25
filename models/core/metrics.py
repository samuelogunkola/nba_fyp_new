import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =========================
# CLASSIFICATION METRICS
# =========================
def classification_metrics(y_true, y_pred, y_proba=None):
    """
    Returns full classification metrics dictionary
    """

    metrics = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["f1"] = f1_score(y_true, y_pred)
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
            metrics["log_loss"] = log_loss(y_true, y_proba)
        except:
            metrics["roc_auc"] = None
            metrics["log_loss"] = None

    return metrics


# =========================
# REGRESSION METRICS
# =========================
def regression_metrics(y_true, y_pred):
    """
    Returns regression metrics dictionary
    """

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


# =========================
# PRETTY PRINT HELPERS
# =========================
def print_classification_results(name, metrics):
    print(f"\n=== {name} ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")

    if "roc_auc" in metrics and metrics["roc_auc"] is not None:
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    if "log_loss" in metrics and metrics["log_loss"] is not None:
        print(f"Log Loss: {metrics['log_loss']:.4f}")

    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])


def print_regression_results(name, metrics):
    print(f"\n=== {name} ===")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R²: {metrics['r2']:.4f}")