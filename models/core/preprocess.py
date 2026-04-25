import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# =========================
# CLEAN FEATURES
# =========================
def clean_features(df, drop_cols=None, remove_win_leakage=True):
    """
    Cleans dataset for modelling:
    - drops unwanted columns
    - keeps numeric only
    - replaces inf with NaN
    - removes win leakage features
    """

    X = df.copy()

    if drop_cols:
        X = X.drop(columns=drop_cols, errors="ignore")

    # numeric only
    X = X.select_dtypes(include=[np.number])

    # replace inf
    X = X.replace([np.inf, -np.inf], np.nan)

    # remove leakage
    if remove_win_leakage:
        leakage_cols = [c for c in X.columns if "win" in c.lower()]
        X = X.drop(columns=leakage_cols, errors="ignore")

    return X


# =========================
# IMPUTER + SCALER CLASS
# =========================
class Preprocessor:
    def __init__(self, scale=True):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler() if scale else None
        self.scale = scale
        self.columns = None

    def fit(self, X):
        self.columns = X.columns

        X_imputed = self.imputer.fit_transform(X)

        if self.scale:
            X_scaled = self.scaler.fit_transform(X_imputed)
            return X_scaled

        return X_imputed

    def transform(self, X):
        X = X[self.columns]  # ensure same column order

        X_imputed = self.imputer.transform(X)

        if self.scale:
            X_scaled = self.scaler.transform(X_imputed)
            return X_scaled

        return X_imputed

    def fit_transform(self, X):
        return self.fit(X)


# =========================
# DROP ALL-NaN TRAIN COLS
# =========================
def drop_all_nan_train_columns(X_train, X_test):
    """
    Drops columns that are entirely NaN in training data
    """
    cols = X_train.columns[X_train.isna().all()].tolist()

    if cols:
        X_train = X_train.drop(columns=cols)
        X_test = X_test.drop(columns=cols)

    return X_train, X_test, cols