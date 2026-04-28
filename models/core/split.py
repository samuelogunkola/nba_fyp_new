# pandas is used to parse dates and work with boolean index masks
import pandas as pd


# splits a dataframe into a training set and a test set based on a cutoff date
# everything before the split date goes into training, everything on or after goes into testing
#
# this is called a time-based split — it's much more realistic than a random split for sports data
# because in real life you always predict future games using only past information
# a random split would let the model accidentally learn from "future" games, which is cheating
def time_split(df, date_col="date", split_date="2019-01-01"):
    """
    Returns boolean masks for a simple time-based split.
    Train: rows before split_date
    Test: rows on/after split_date
    """

    # make sure the date column actually exists before doing anything
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in dataframe.")

    # parse the date column into proper datetime objects so we can compare them
    # this handles cases where the dates are stored as strings in the CSV
    dates = pd.to_datetime(df[date_col])

    # create boolean masks — True for rows that belong in each set
    # train gets every game that happened strictly before the split date
    train_idx = dates < pd.to_datetime(split_date)

    # test gets every game on or after the split date
    test_idx = dates >= pd.to_datetime(split_date)

    # safety check: if the split date is set wrong the training set could be empty
    # an empty training set would mean the model has nothing to learn from
    if train_idx.sum() == 0:
        raise ValueError("Time split produced empty training set.")

    # similarly, an empty test set means we have nothing to evaluate the model on
    if test_idx.sum() == 0:
        raise ValueError("Time split produced empty test set.")

    # return the two boolean masks — the caller uses them to slice X and y
    return train_idx, test_idx


# uses the boolean masks produced by time_split (or season_holdout_split) to
# slice both the feature matrix X and the target series y into train and test halves
def split_X_y(X, y, train_idx, test_idx):
    """
    Splits features and target using boolean masks.
    """

    # .loc[mask] selects only the rows where the mask is True
    # .copy() ensures we get independent copies, not views into the original dataframe
    # modifying X_train later won't accidentally change X as a result
    X_train = X.loc[train_idx].copy()
    X_test = X.loc[test_idx].copy()

    # do the same for the target column
    y_train = y.loc[train_idx].copy()
    y_test = y.loc[test_idx].copy()

    return X_train, X_test, y_train, y_test


# like split_X_y but handles multiple target columns at once
# useful for player stat prediction where we want to predict pts, reb, and ast simultaneously
# y_dict is a dictionary mapping a name (e.g. "pts") to a target series (a pandas Series)
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

    # split the feature matrix the same way as in split_X_y
    X_train = X.loc[train_idx].copy()
    X_test = X.loc[test_idx].copy()

    # split each target series in the dictionary using a dictionary comprehension
    # the result is still a dictionary, just with each series sliced to the right rows
    y_train = {name: y.loc[train_idx].copy() for name, y in y_dict.items()}
    y_test = {name: y.loc[test_idx].copy() for name, y in y_dict.items()}

    return X_train, X_test, y_train, y_test


# an alternative splitting strategy: instead of splitting by date, hold out an entire season
# this is useful for evaluating whether the model generalises to a completely unseen season
# e.g. train on 2012–2018, test on 2019 — a very rigorous test of generalisation
def season_holdout_split(df, season_col="season", test_season=None):
    """
    Hold out a full season for testing.
    Example:
    - train on all seasons except 2019
    - test on season 2019
    """

    # make sure the season column exists in the dataframe
    if season_col not in df.columns:
        raise ValueError(f"Column '{season_col}' not found in dataframe.")

    # the caller must specify which season to hold out — there's no sensible default
    if test_season is None:
        raise ValueError("test_season must be provided.")

    # rows from any season other than the test season go into training
    train_idx = df[season_col] != test_season

    # rows from the test season are held out for evaluation
    test_idx = df[season_col] == test_season

    # guard against a badly specified season value leaving one set empty
    if train_idx.sum() == 0:
        raise ValueError("Season holdout produced empty training set.")
    if test_idx.sum() == 0:
        raise ValueError("Season holdout produced empty test set.")

    return train_idx, test_idx


# generates a series of walk-forward splits — each split trains on all seasons seen so far
# and tests on the very next season, then the window expands by one and repeats
#
# this is also called "expanding window" cross-validation and it's the gold standard
# for evaluating time-series models because it perfectly mimics real-world conditions
# where you always train on the past and predict the future, one step at a time
def rolling_window_splits(
    df,
    season_col="season",
    min_train_seasons=3    # don't start generating splits until we have at least this many training seasons
):
    """
    Generates walk-forward season-based splits.

    Example:
    if seasons are [2012, 2013, 2014, 2015, 2016]
    and min_train_seasons=3, yields:
      train=[2012,2013,2014], test=[2015]
      train=[2012,2013,2014,2015], test=[2016]
    """

    # make sure the season column exists before going any further
    if season_col not in df.columns:
        raise ValueError(f"Column '{season_col}' not found in dataframe.")

    # get a sorted list of every unique season in the dataset
    # sorted() ensures the seasons are always in chronological order
    seasons = sorted(df[season_col].dropna().unique().tolist())

    # we need more seasons than the minimum training window, otherwise there's nothing left to test on
    if len(seasons) <= min_train_seasons:
        raise ValueError(
            f"Not enough seasons for rolling splits. "
            f"Need > {min_train_seasons}, got {len(seasons)}"
        )

    # we'll collect all the generated splits into this list before returning them
    splits = []

    # start the loop at index min_train_seasons so we always have enough seasons to train on
    # each iteration uses all seasons up to i for training, and season i for testing
    for i in range(min_train_seasons, len(seasons)):
        # all seasons before position i form the training window
        train_seasons = seasons[:i]

        # the season at position i is held out as the test season
        test_season = seasons[i]

        # create boolean masks the same way as the other split functions
        train_idx = df[season_col].isin(train_seasons)
        test_idx = df[season_col] == test_season

        # skip this split if either set came out empty — shouldn't happen but we check anyway
        if train_idx.sum() == 0 or test_idx.sum() == 0:
            continue

        # store this split as a dictionary so the caller knows exactly which seasons were used
        splits.append({
            "train_seasons": train_seasons,
            "test_season": test_season,
            "train_idx": train_idx,
            "test_idx": test_idx
        })

    # return all the splits — the caller loops over them to run an experiment for each one
    return splits
