import pandas as pd


def time_split(df, date_col="date", split_date="2019-01-01"):
    """
    Returns boolean masks for a simple time-based split.
    Train: rows before split_date
    Test: rows on/after split_date
    """
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in dataframe.")

    dates = pd.to_datetime(df[date_col])

    train_idx = dates < pd.to_datetime(split_date)
    test_idx = dates >= pd.to_datetime(split_date)

    if train_idx.sum() == 0:
        raise ValueError("Time split produced empty training set.")
    if test_idx.sum() == 0:
        raise ValueError("Time split produced empty test set.")

    return train_idx, test_idx


def split_X_y(X, y, train_idx, test_idx):
    """
    Splits features and target using boolean masks.
    """
    X_train = X.loc[train_idx].copy()
    X_test = X.loc[test_idx].copy()

    y_train = y.loc[train_idx].copy()
    y_test = y.loc[test_idx].copy()

    return X_train, X_test, y_train, y_test


def multi_target_split(X, y_dict, train_idx, test_idx):
    """
    Splits features and multiple targets using boolean masks.

    y_dict example:
    {
        "pts": y_pts,
        "ast": y_ast,
        "reb": y_reb
    }
    """
    X_train = X.loc[train_idx].copy()
    X_test = X.loc[test_idx].copy()

    y_train = {name: y.loc[train_idx].copy() for name, y in y_dict.items()}
    y_test = {name: y.loc[test_idx].copy() for name, y in y_dict.items()}

    return X_train, X_test, y_train, y_test


def season_holdout_split(df, season_col="season", test_season=None):
    """
    Hold out a full season for testing.
    Example:
    - train on all seasons except 2019
    - test on season 2019
    """
    if season_col not in df.columns:
        raise ValueError(f"Column '{season_col}' not found in dataframe.")

    if test_season is None:
        raise ValueError("test_season must be provided.")

    train_idx = df[season_col] != test_season
    test_idx = df[season_col] == test_season

    if train_idx.sum() == 0:
        raise ValueError("Season holdout produced empty training set.")
    if test_idx.sum() == 0:
        raise ValueError("Season holdout produced empty test set.")

    return train_idx, test_idx


def rolling_window_splits(
    df,
    season_col="season",
    min_train_seasons=3
):
    """
    Generates walk-forward season-based splits.

    Example:
    if seasons are [2012, 2013, 2014, 2015, 2016]
    and min_train_seasons=3, yields:
      train=[2012,2013,2014], test=[2015]
      train=[2012,2013,2014,2015], test=[2016]
    """
    if season_col not in df.columns:
        raise ValueError(f"Column '{season_col}' not found in dataframe.")

    seasons = sorted(df[season_col].dropna().unique().tolist())

    if len(seasons) <= min_train_seasons:
        raise ValueError(
            f"Not enough seasons for rolling splits. "
            f"Need > {min_train_seasons}, got {len(seasons)}"
        )

    splits = []

    for i in range(min_train_seasons, len(seasons)):
        train_seasons = seasons[:i]
        test_season = seasons[i]

        train_idx = df[season_col].isin(train_seasons)
        test_idx = df[season_col] == test_season

        if train_idx.sum() == 0 or test_idx.sum() == 0:
            continue

        splits.append({
            "train_seasons": train_seasons,
            "test_season": test_season,
            "train_idx": train_idx,
            "test_idx": test_idx
        })

    return splits