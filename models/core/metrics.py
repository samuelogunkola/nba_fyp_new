# numpy is used here for the square root calculation needed to compute RMSE
import numpy as np

# sklearn.metrics contains ready-made implementations of all standard evaluation functions
# so we don't have to write the maths ourselves
from sklearn.metrics import (
    accuracy_score,       # percentage of predictions that were exactly correct
    f1_score,             # harmonic mean of precision and recall — best single classification metric
    roc_auc_score,        # how well the model separates the two classes across all thresholds
    log_loss,             # measures the uncertainty of probability predictions (lower is better)
    confusion_matrix,     # a 2x2 table showing true positives, false positives, etc.
    mean_absolute_error,  # average prediction error in the same units as the target (e.g. points)
    mean_squared_error,   # like MAE but squares errors first, so large mistakes are penalised more
    r2_score              # proportion of variance the model explains — 1.0 is perfect, 0.0 is just the mean
)


# computes all relevant classification metrics given the true labels and model predictions
# y_true = the real outcomes (e.g. 1 for home win, 0 for away win)
# y_pred = the model's hard class predictions (also 0 or 1)
# y_proba = the model's probability estimates for the positive class (optional, needed for ROC-AUC)
def classification_metrics(y_true, y_pred, y_proba=None):
    """
    Returns full classification metrics dictionary
    """

    # start with an empty dictionary and add metrics one by one
    metrics = {}

    # accuracy: out of all predictions, what fraction were correct?
    # e.g. 0.68 means the model got 68% of games right
    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    # F1 score: balances precision (did we predict wins accurately?) and
    # recall (did we catch all the actual wins?) into a single number
    # more useful than accuracy when one outcome is more common than the other
    metrics["f1"] = f1_score(y_true, y_pred)

    # confusion matrix: a 2x2 grid showing:
    # [[true negatives, false positives],
    #  [false negatives, true positives]]
    # useful for seeing exactly what kind of mistakes the model makes
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    # ROC-AUC and log loss both require probability estimates, not just hard predictions
    # so we only compute them if y_proba was passed in
    if y_proba is not None:
        try:
            # ROC-AUC ranges from 0.5 (random guessing) to 1.0 (perfect separation)
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

            # log loss penalises confident wrong predictions more heavily than uncertain ones
            # lower log loss = the model's probabilities are well calibrated
            metrics["log_loss"] = log_loss(y_true, y_proba)
        except:
            # if the calculation fails for any reason (e.g. only one class in the test set),
            # store None rather than crashing the whole experiment
            metrics["roc_auc"] = None
            metrics["log_loss"] = None

    return metrics


# computes all standard regression metrics given the true values and model predictions
# y_true = the actual numbers (e.g. real points scored)
# y_pred = what the model predicted
def regression_metrics(y_true, y_pred):
    """
    Returns regression metrics dictionary
    """

    # MAE: mean absolute error — on average, how many units off is the prediction?
    # e.g. MAE of 9.5 means the model is off by ~9.5 points on average
    # easy to interpret because it's in the same units as the target
    mae = mean_absolute_error(y_true, y_pred)

    # RMSE: root mean squared error — like MAE but squares each error before averaging
    # this means large mistakes are punished more harshly
    # we take the square root at the end to bring it back to the original units
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # R²: coefficient of determination — how much of the variation in the target does the model explain?
    # 1.0 = perfect, 0.0 = no better than just predicting the average every time, negative = worse than average
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


# prints a formatted summary of classification results to the console
# called after every classification experiment so we can see the numbers while the script runs
def print_classification_results(name, metrics):
    # print the model name as a header so it's easy to find in long console output
    print(f"\n=== {name} ===")

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")

    # ROC-AUC and log loss are only printed if they were successfully computed
    if "roc_auc" in metrics and metrics["roc_auc"] is not None:
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    if "log_loss" in metrics and metrics["log_loss"] is not None:
        print(f"Log Loss: {metrics['log_loss']:.4f}")

    # print the confusion matrix on its own line so the grid layout is readable
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])


# prints a formatted summary of regression results to the console
# called after every regression experiment
def print_regression_results(name, metrics):
    print(f"\n=== {name} ===")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R²: {metrics['r2']:.4f}")
