# argparse lets us pass command-line arguments to the script (like --team-master and --output)
import argparse

# Path gives us OS-independent file path handling
from pathlib import Path

# numpy gives us maths tools — used here to handle infinity values
import numpy as np

# pandas is used to load, transform, and save the dataset
import pandas as pd


# the rolling window sizes to use when computing rolling averages and standard deviations
# 3-game window captures very recent form; 5-game window captures slightly longer trends
ROLL_WINDOWS = [3, 5]


# load a dataset from either a CSV or parquet file
# parquet is much faster to read/write than CSV for large files, which is why we support both
def load_data(path):
    path = Path(path)

    if path.suffix == ".csv":
        return pd.read_csv(path)

    if path.suffix == ".parquet":
        return pd.read_parquet(path)

    # if the file extension is neither .csv nor .parquet, stop and tell the caller
    raise ValueError("Unsupported file format. Use .csv or .parquet")


# save a dataframe to disk in either CSV or parquet format
# also creates any missing parent folders so the save never fails due to a missing directory
def save_data(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".csv":
        df.to_csv(path, index=False)

    elif path.suffix == ".parquet":
        df.to_parquet(path, index=False)

    else:
        raise ValueError("Unsupported output format. Use .csv or .parquet")


# basic cleaning step that runs before any feature engineering
# parses dates, and sorts by team then date so rolling calculations go in the right order
def preprocess(df):
    df = df.copy()

    # parse the date column into datetime objects — rolling windows rely on rows being in date order
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # sort by team first, then date, then gameid to guarantee chronological order within each team
    df = df.sort_values(["team", "date", "gameid"]).reset_index(drop=True)

    return df


# estimate the number of possessions in a game using a standard basketball formula
# possessions = field goal attempts + 0.44 * free throw attempts - offensive rebounds + turnovers
# the 0.44 coefficient is a well-known approximation for how many possessions a free throw uses
# this is needed to compute offensive and defensive rating (points per 100 possessions)
def add_possessions(df):
    df = df.copy()

    df["possessions"] = (
        df["fga"]
        + 0.44 * df["fta"]
        - df["oreb"]
        + df["tov"]
    )

    return df


# compute how many days each team had between games, and flag back-to-back situations
# fatigue is a known factor in NBA performance, especially on the second night of a back-to-back
def add_rest_features(df):
    df = df.copy()

    # sort by team and date so .diff() always compares consecutive games for the same team
    df = df.sort_values(["team", "date"])

    # .diff() on a datetime column gives the gap between consecutive rows per team
    # .dt.days converts the timedelta to a plain integer number of days
    df["days_since_last_game"] = (
        df.groupby("team")["date"]
        .diff()
        .dt.days
    )

    # a back-to-back is when the gap between games is exactly 1 day
    # cast to int so it's 1 for back-to-back and 0 otherwise — easier for the model to use
    df["back_to_back"] = (
        df["days_since_last_game"] == 1
    ).astype(int)

    # fill NaN for the first game of each team's history — assume 7 days rest as a neutral default
    df["days_since_last_game"] = df["days_since_last_game"].fillna(7)

    return df


# attach the opponent's points and possessions to each team's row
# this is needed to compute defensive rating (how many points the opponent scored per 100 possessions)
def add_opponent_stats(df):
    df = df.copy()

    # build a small lookup table: gameid → opponent's team name, points, and possessions
    opp = df[
        [
            "gameid",
            "team",
            "pts",
            "possessions",
        ]
    ].copy()

    # rename columns so when we merge them back they're clearly labelled as opponent stats
    opp = opp.rename(
        columns={
            "team": "opp_team",
            "pts": "opp_pts",
            "possessions": "opp_possessions",
        }
    )

    # merge the opponent stats onto the main dataframe, joining on gameid
    # this gives every row both the team's own stats and the opponent's stats for that game
    df = df.merge(opp, on="gameid", how="left")

    # after the merge each game appears twice (once per team), each with the other team as opponent
    # remove any rows where a team is matched against itself (can happen if data is messy)
    df = df[df["team"] != df["opp_team"]].copy()

    return df


# helper function that divides two series but avoids dividing by zero
# replacing 0 with NaN means the result is NaN rather than infinity
def safe_divide(a, b):
    return a / b.replace(0, np.nan)


# compute advanced efficiency metrics for each team in each game
# these give a better picture of how efficient a team was than raw box-score numbers
def add_advanced_metrics(df):
    df = df.copy()

    # offensive rating: how many points the team scored per 100 possessions
    # multiplying by 100 makes it easier to interpret (e.g. 112 = 112 points per 100 possessions)
    df["off_rating"] = safe_divide(df["pts"], df["possessions"]) * 100

    # defensive rating: how many points the opponent scored per 100 possessions against this team
    # lower is better — a team that holds opponents to fewer points has a strong defence
    df["def_rating"] = safe_divide(df["opp_pts"], df["opp_possessions"]) * 100

    # net rating: offensive minus defensive rating — the team's overall efficiency margin
    # positive means they outscored opponents on a per-possession basis; negative means they didn't
    df["net_rating"] = df["off_rating"] - df["def_rating"]

    # true shooting proxy: measures how efficiently the team scored relative to their shot attempts
    # it accounts for the fact that 3-pointers and free throws have different values
    # the 0.44 factor is the same approximation used in the possessions formula
    df["ts_proxy"] = safe_divide(
        df["pts"],
        2 * (df["fga"] + 0.44 * df["fta"])
    )

    return df


# create rolling average and rolling standard deviation features for each team
# rolling features capture recent form without using any information from the current game
# we use .shift(1) to make sure the current game's stats are never included in the window
# so all rolling features are genuinely pregame-safe
def add_rolling_features(df):
    df = df.copy()

    # sort by team and date first so the rolling windows always look backwards in time
    df = df.sort_values(["team", "date", "gameid"])

    # the columns we want to create rolling features for
    rolling_cols = [
        "pts",
        "reb",
        "ast",
        "tov",
        "stl",
        "blk",
        "possessions",
        "off_rating",
        "def_rating",
        "net_rating",
        "ts_proxy",
        "days_since_last_game",
        "back_to_back",
    ]

    # include win/loss history if the dataset has a win column
    if "win" in df.columns:
        rolling_cols.append("win")

    for col in rolling_cols:
        # skip if this column doesn't exist in the dataframe (not all datasets have every stat)
        if col not in df.columns:
            continue

        for window in ROLL_WINDOWS:
            # rolling mean: average value over the last N games (excluding the current game)
            # groupby("team") ensures each team's window is computed separately
            # .shift(1) moves the window back one game so the current game is never included
            # min_periods=1 means we compute a mean even if we only have 1 game in the window
            df[f"{col}_roll_mean_{window}"] = (
                df.groupby("team")[col]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            )

            # rolling std: how consistent the team has been over the last N games
            # high std = unpredictable; low std = consistent performer
            # min_periods=2 because standard deviation needs at least 2 values to be meaningful
            df[f"{col}_roll_std_{window}"] = (
                df.groupby("team")[col]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=2).std())
            )

    return df


# track how many games each team has played so far in the season
# early-season games have less rolling history so this helps the model know how much to trust the features
def add_team_game_number(df):
    df = df.copy()

    if "season" in df.columns:
        # if we have a season column, reset the counter at the start of each new season
        df["team_game_number"] = (
            df.groupby(["team", "season"])
            .cumcount()
            + 1   # cumcount() starts at 0 so we add 1 to make it 1-indexed
        )
    else:
        # if there's no season column, just count games since the beginning of the dataset
        df["team_game_number"] = (
            df.groupby("team")
            .cumcount()
            + 1
        )

    return df


# runs all feature engineering steps in the correct order and returns a fully featured team dataframe
# this is the main pipeline function — call this to go from raw box scores to a model-ready dataset
def build_team_features(df):
    df = preprocess(df)           # parse dates and sort rows
    df = add_possessions(df)      # estimate possessions per game
    df = add_rest_features(df)    # days of rest and back-to-back flags
    df = add_opponent_stats(df)   # attach opponent points and possessions
    df = add_advanced_metrics(df) # compute offensive/defensive/net rating and true shooting
    df = add_rolling_features(df) # rolling averages and standard deviations for all stats
    df = add_team_game_number(df) # game count within the season

    # replace any infinity values that crept in during division operations with NaN
    # the imputer downstream can handle NaN but not infinity
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


# removes all columns that would only be known after a game ends
# this is the leakage prevention step — everything in this list is a same-game result
# keeping any of these as features would give the model an unfair advantage during training
def remove_leakage_cols(df):
    """
    Removes same-game box score columns.

    We only keep pre-game-safe features:
    - rolling means/stds
    - rest features
    - team metadata needed for merge
    """

    leakage_cols = [
        # the actual points scored — obvious leakage for score and win prediction
        "pts",
        "opp_pts",
        "plus_minus",

        # raw box score stats — all only available after the game ends
        "fgm",
        "fga",
        "fgpct",
        "3pm",
        "3pa",
        "3ppct",
        "ftm",
        "fta",
        "ftpct",
        "oreb",
        "dreb",
        "reb",
        "ast",
        "tov",
        "stl",
        "blk",
        "pf",
        "possessions",
        "opp_possessions",

        # advanced metrics computed from same-game box scores — also leakage
        "off_rating",
        "def_rating",
        "net_rating",
        "ts_proxy",

        # the actual game outcome — the most obvious leakage of all
        "win",
    ]

    # drop any listed columns that exist, ignore any that don't
    return df.drop(columns=leakage_cols, errors="ignore")


# after removing leakage columns, keep only the columns that are safe to use as model features
# this includes identifier columns (gameid, date, team, home, away) plus rolling averages
def keep_numeric_pregame_features(df):
    # always keep these identifier and metadata columns
    keep_cols = [
        "gameid",
        "date",
        "season",
        "team",
        "home",
        "away",
    ]

    # only include identifier columns that actually exist in this dataframe
    keep_cols = [c for c in keep_cols if c in df.columns]

    # find all numeric columns that survived the leakage removal step
    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    # from the numeric columns, keep only rolling features and a few safe non-rolling columns
    # rolling features are the core pregame signals — they capture recent team form
    # days_since_last_game, back_to_back, and team_game_number are also pregame-safe
    rolling_cols = [
        c for c in numeric_cols
        if "_roll_" in c
        or c in [
            "days_since_last_game",
            "back_to_back",
            "team_game_number",
        ]
    ]

    # combine identifiers and rolling feature columns into the final column list
    final_cols = keep_cols + rolling_cols

    return df[final_cols].copy()


# builds the final matchup-level dataset by joining home and away team feature rows
# the output has one row per game with all home team features and all away team features side by side
# this is the format the win and score prediction models expect as input
def build_matchups(df):
    df = df.copy()

    # make sure the required columns are present before doing anything
    if not {"home", "away", "team"}.issubset(df.columns):
        raise ValueError("Dataset must contain 'team', 'home', and 'away' columns.")

    # split the team-level dataframe into two separate views: home team rows and away team rows
    home_df = df[df["team"] == df["home"]].copy()
    away_df = df[df["team"] == df["away"]].copy()

    # save the actual scores now, before we remove them as leakage
    # we'll add them back at the very end as the prediction targets
    targets = home_df[["gameid", "pts"]].rename(
        columns={"pts": "home_pts"}
    )

    # merge in the away team's actual score to get both scores in the same row
    targets = targets.merge(
        away_df[["gameid", "pts"]].rename(
            columns={"pts": "away_pts"}
        ),
        on="gameid",
        how="inner",
    )

    # compute point spread (positive = home team winning by that many points)
    targets["point_spread"] = (
        targets["home_pts"] - targets["away_pts"]
    )

    # compute total points (the combined score of both teams — the "over/under")
    targets["total_points"] = (
        targets["home_pts"] + targets["away_pts"]
    )

    # now strip out all the same-game leakage columns from both views
    home_df = remove_leakage_cols(home_df)
    away_df = remove_leakage_cols(away_df)

    # keep only the safe pregame rolling features (and identifier columns)
    home_df = keep_numeric_pregame_features(home_df)
    away_df = keep_numeric_pregame_features(away_df)

    # add a "home_" prefix to every home team column and "away_" to every away team column
    # this avoids column name collisions when we join the two views together
    home_df = home_df.add_prefix("home_")
    away_df = away_df.add_prefix("away_")

    # join the home and away rows into one wide row per game, matching on gameid
    matchups = home_df.merge(
        away_df,
        left_on="home_gameid",
        right_on="away_gameid",
        how="inner",
    )

    # tidy up: rename the duplicated gameid column and drop the now-redundant away version
    matchups = matchups.rename(columns={"home_gameid": "gameid"})
    matchups = matchups.drop(columns=["away_gameid"], errors="ignore")

    # restore clean date and season columns by renaming from the prefixed versions
    if "home_date" in matchups.columns:
        matchups = matchups.rename(columns={"home_date": "date"})

    if "home_season" in matchups.columns:
        matchups = matchups.rename(columns={"home_season": "season"})

    # drop the remaining duplicate and redundant columns created by the prefix step
    matchups = matchups.drop(
        columns=[
            "away_date",
            "away_season",
            "home_team",
            "away_team",
            "home_home",
            "away_home",
            "home_away",
            "away_away",
        ],
        errors="ignore",
    )

    # add matchup interaction features — the difference between home and away rolling stats
    # these give the model a direct head-to-head comparison rather than two separate values
    def add_diff(home_col, away_col, output_col):
        # only create the diff column if both input columns exist
        if home_col in matchups.columns and away_col in matchups.columns:
            matchups[output_col] = matchups[home_col] - matchups[away_col]

    # net rating diff: positive = home team has been more efficient recently
    add_diff(
        "home_net_rating_roll_mean_5",
        "away_net_rating_roll_mean_5",
        "net_rating_diff",
    )

    # home offence vs away defence — does the home team score better than the away team defends?
    add_diff(
        "home_off_rating_roll_mean_5",
        "away_def_rating_roll_mean_5",
        "home_off_vs_away_def",
    )

    # away offence vs home defence — does the away team score better than the home team defends?
    add_diff(
        "away_off_rating_roll_mean_5",
        "home_def_rating_roll_mean_5",
        "away_off_vs_home_def",
    )

    # pace diff: positive = home team plays at a faster tempo than the away team
    add_diff(
        "home_possessions_roll_mean_5",
        "away_possessions_roll_mean_5",
        "pace_diff",
    )

    # expected pace: the average of both teams' recent tempos — a proxy for how fast the game will be
    # faster pace games tend to have higher total scores, which is useful for score prediction
    if {
        "home_possessions_roll_mean_5",
        "away_possessions_roll_mean_5",
    }.issubset(matchups.columns):
        matchups["expected_pace"] = (
            matchups["home_possessions_roll_mean_5"]
            + matchups["away_possessions_roll_mean_5"]
        ) / 2

    # merge the target scores back in now that all features are assembled
    # these are the labels the models will be trained to predict
    matchups = matchups.merge(targets, on="gameid", how="inner")

    # replace any infinity values that crept in during feature computation
    matchups = matchups.replace([np.inf, -np.inf], np.nan)

    # sort by date so the dataset is in chronological order for the time-based train/test split
    if "date" in matchups.columns:
        matchups = matchups.sort_values("date").reset_index(drop=True)

    return matchups


# entry point when running this script from the command line
# takes the raw team master file and outputs a fully processed matchup dataset
def main():
    # set up command-line argument parsing
    parser = argparse.ArgumentParser()

    # path to the input team-level dataset (box scores per team per game)
    parser.add_argument(
        "--team-master",
        required=True,
        help="Path to team_master.csv or team_master.parquet",
    )

    # path where the final matchup dataset should be saved
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save score_matchups.parquet or .csv",
    )

    args = parser.parse_args()

    # load the raw team master file
    team_master = load_data(args.team_master)
    print(f"Loaded: {team_master.shape}")

    # run all feature engineering steps to produce team-level rolling features
    team_features = build_team_features(team_master)
    print(f"Features built: {team_features.shape}")

    # join home and away rows into one row per game — the final model-ready format
    score_matchups = build_matchups(team_features)
    print(f"Matchups built: {score_matchups.shape}")

    # final safety check: scan for any remaining columns that look like same-game leakage
    # these are columns that end with box-score suffixes but aren't the intended target columns
    suspicious_cols = [
        c for c in score_matchups.columns
        if (
            c not in ["home_pts", "away_pts", "point_spread", "total_points"]
            and (
                c.endswith("_pts")
                or "opp_pts" in c
                or "plus_minus" in c
                or c.endswith("_reb")
                or c.endswith("_ast")
                or c.endswith("_tov")
                or c.endswith("_stl")
                or c.endswith("_blk")
                or c.endswith("_fga")
                or c.endswith("_fta")
                or c.endswith("_fgm")
                or c.endswith("_ftm")
            )
        )
    ]

    # warn loudly if any suspicious columns survived — these should be investigated before training
    if suspicious_cols:
        print("\nWARNING: Possible leakage columns found:")
        for col in suspicious_cols:
            print("-", col)
    else:
        print("Leakage safety check passed.")

    # save the final dataset to disk
    save_data(score_matchups, args.output)
    print(f"Saved to: {args.output}")


# only run main() when this script is executed directly from the command line
if __name__ == "__main__":
    main()
