# pandas is used to work with dataframes (tables of data)
import pandas as pd

# numpy gives us access to numeric type checks and special values like infinity and NaN
import numpy as np

# SimpleImputer fills in missing values automatically — we use the median strategy,
# meaning each missing cell gets replaced with the median of that column in the training set
from sklearn.impute import SimpleImputer

# StandardScaler rescales each feature so it has a mean of 0 and a standard deviation of 1
# some models (like Ridge) perform better when features are on a similar scale
from sklearn.preprocessing import StandardScaler


# takes a raw dataframe and strips out everything the model shouldn't see
# returns a clean feature matrix X ready for training or inference
def clean_features(df, drop_cols=None, remove_win_leakage=True):
    """
    Cleans dataset for modelling:
    - drops unwanted columns
    - keeps numeric only
    - replaces inf with NaN
    - removes win leakage features
    """

    # make a copy so we never accidentally modify the original dataframe
    X = df.copy()

    # drop any columns the caller explicitly said to remove
    # errors="ignore" means it won't crash if a listed column doesn't exist
    if drop_cols:
        X = X.drop(columns=drop_cols, errors="ignore")

    # keep only numeric columns — models can't work with raw strings or dates
    # this also automatically removes things like team names and game IDs
    X = X.select_dtypes(include=[np.number])

    # replace positive and negative infinity with NaN
    # infinity can appear in rolling calculations if there's a division by zero somewhere
    # the imputer can fill NaN but can't handle infinity, so we convert it here
    X = X.replace([np.inf, -np.inf], np.nan)

    # if win leakage removal is turned on, drop any column whose name contains "win"
    # these columns reveal the outcome of the game and would give the model an unfair advantage
    # e.g. "home_win", "away_win_roll_mean_5" — these are only known after the game ends
    if remove_win_leakage:
        leakage_cols = [c for c in X.columns if "win" in c.lower()]
        X = X.drop(columns=leakage_cols, errors="ignore")

    return X


# this class wraps the imputer and optional scaler into a single reusable object
# having both steps in one class makes it easier to apply the same transformations
# consistently to both the training set and the test set
class Preprocessor:
    def __init__(self, scale=True):
        # use the median to fill missing values — median is more robust than the mean
        # because it isn't pulled by extreme outliers the way the mean is
        self.imputer = SimpleImputer(strategy="median")

        # only create a scaler if scale=True was passed in
        # scaling is useful for linear models but unnecessary for tree-based ones
        self.scaler = StandardScaler() if scale else None

        self.scale = scale

        # we'll store the column names after the first fit so we can enforce the same
        # order when transforming the test set later
        self.columns = None

    def fit(self, X):
        # remember the exact column names and order from the training set
        # this lets us reorder test set columns to match before transforming
        self.columns = X.columns

        # fit the imputer on the training data and immediately apply it
        # fit_transform learns the median of each column then fills NaNs in one step
        X_imputed = self.imputer.fit_transform(X)

        if self.scale:
            # fit the scaler on the imputed training data and apply it
            # StandardScaler subtracts the mean and divides by the standard deviation
            X_scaled = self.scaler.fit_transform(X_imputed)
            return X_scaled

        # if no scaling was requested, just return the imputed array
        return X_imputed

    def transform(self, X):
        # reorder the test set columns to match the training set column order
        # this prevents a mismatch if columns arrived in a different order
        X = X[self.columns]

        # apply the already-fitted imputer to the test data
        # importantly we use .transform() not .fit_transform() here —
        # we want to fill using the medians learned from training, not from the test set
        X_imputed = self.imputer.transform(X)

        if self.scale:
            # apply the already-fitted scaler using training set statistics
            # again, .transform() not .fit_transform() so test data doesn't influence the scaling
            X_scaled = self.scaler.transform(X_imputed)
            return X_scaled

        return X_imputed

    def fit_transform(self, X):
        # convenience method that just calls fit() — fit() already returns the transformed result
        # having this method means the class works seamlessly with scikit-learn pipelines
        return self.fit(X)


# removes columns that are completely empty (all NaN) in the training set
# a column with no values at all can't teach the model anything useful
# and some scikit-learn models will error or warn if they see all-NaN columns
def drop_all_nan_train_columns(X_train, X_test):
    """
    Drops columns that are entirely NaN in training data
    """

    # find the names of any columns where every single value in the training set is NaN
    cols = X_train.columns[X_train.isna().all()].tolist()

    if cols:
        # drop those columns from both sets so they always have matching shapes
        X_train = X_train.drop(columns=cols)
        X_test = X_test.drop(columns=cols)

    # return the cleaned dataframes and the list of dropped column names
    # the caller logs the dropped names so we know what was removed
    return X_train, X_test, cols
