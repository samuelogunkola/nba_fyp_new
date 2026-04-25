import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROLL_WINDOWS = [3, 5]


# ============================================================
# LOAD / SAVE
# ============================================================

def load_data(path):
    path = Path(path)

    if path.suffix == ".csv":
        return pd.read_csv(path)

    if path.suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError("Unsupported file format. Use .csv or .parquet")


def save_data(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".csv":
        df.to_csv(path, index=False)

    elif path.suffix == ".parquet":
        df.to_parquet(path, index=False)

    else:
        raise ValueError("Unsupported output format. Use .csv or .parquet")


# ============================================================
# BASIC CLEANING
# ============================================================

def preprocess(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["team", "date", "gameid"]).reset_index(drop=True)
    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_possessions(df):
    df = df.copy()

    df["possessions"] = (
        df["fga"]
        + 0.44 * df["fta"]
        - df["oreb"]
        + df["tov"]
    )

    return df


def add_rest_features(df):
    df = df.copy()
    df = df.sort_values(["team", "date"])

    df["days_since_last_game"] = (
        df.groupby("team")["date"]
        .diff()
        .dt.days
    )

    df["back_to_back"] = (
        df["days_since_last_game"] == 1
    ).astype(int)

    df["days_since_last_game"] = df["days_since_last_game"].fillna(7)

    return df


def add_opponent_stats(df):
    df = df.copy()

    opp = df[
        [
            "gameid",
            "team",
            "pts",
            "possessions",
        ]
    ].copy()

    opp = opp.rename(
        columns={
            "team": "opp_team",
            "pts": "opp_pts",
            "possessions": "opp_possessions",
        }
    )

    df = df.merge(opp, on="gameid", how="left")
    df = df[df["team"] != df["opp_team"]].copy()

    return df


def safe_divide(a, b):
    return a / b.replace(0, np.nan)


def add_advanced_metrics(df):
    df = df.copy()

    df["off_rating"] = safe_divide(df["pts"], df["possessions"]) * 100
    df["def_rating"] = safe_divide(df["opp_pts"], df["opp_possessions"]) * 100
    df["net_rating"] = df["off_rating"] - df["def_rating"]

    df["ts_proxy"] = safe_divide(
        df["pts"],
        2 * (df["fga"] + 0.44 * df["fta"])
    )

    return df


def add_rolling_features(df):
    df = df.copy()
    df = df.sort_values(["team", "date", "gameid"])

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

    if "win" in df.columns:
        rolling_cols.append("win")

    for col in rolling_cols:
        if col not in df.columns:
            continue

        for window in ROLL_WINDOWS:
            df[f"{col}_roll_mean_{window}"] = (
                df.groupby("team")[col]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            )

            df[f"{col}_roll_std_{window}"] = (
                df.groupby("team")[col]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=2).std())
            )

    return df


def add_team_game_number(df):
    df = df.copy()

    if "season" in df.columns:
        df["team_game_number"] = (
            df.groupby(["team", "season"])
            .cumcount()
            + 1
        )
    else:
        df["team_game_number"] = (
            df.groupby("team")
            .cumcount()
            + 1
        )

    return df


def build_team_features(df):
    df = preprocess(df)
    df = add_possessions(df)
    df = add_rest_features(df)
    df = add_opponent_stats(df)
    df = add_advanced_metrics(df)
    df = add_rolling_features(df)
    df = add_team_game_number(df)

    df = df.replace([np.inf, -np.inf], np.nan)

    return df


# ============================================================
# LEAKAGE REMOVAL
# ============================================================

def remove_leakage_cols(df):
    """
    Removes same-game box score columns.

    We only keep pre-game-safe features:
    - rolling means/stds
    - rest features
    - team metadata needed for merge
    """

    leakage_cols = [
        # direct score targets
        "pts",
        "opp_pts",
        "plus_minus",

        # same-game box score stats
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

        # same-game advanced metrics
        "off_rating",
        "def_rating",
        "net_rating",
        "ts_proxy",

        # same-game result
        "win",
    ]

    return df.drop(columns=leakage_cols, errors="ignore")


def keep_numeric_pregame_features(df):
    keep_cols = [
        "gameid",
        "date",
        "season",
        "team",
        "home",
        "away",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]

    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    rolling_cols = [
        c for c in numeric_cols
        if "_roll_" in c
        or c in [
            "days_since_last_game",
            "back_to_back",
            "team_game_number",
        ]
    ]

    final_cols = keep_cols + rolling_cols

    return df[final_cols].copy()


# ============================================================
# MATCHUP DATASET
# ============================================================

def build_matchups(df):
    df = df.copy()

    if not {"home", "away", "team"}.issubset(df.columns):
        raise ValueError("Dataset must contain 'team', 'home', and 'away' columns.")

    home_df = df[df["team"] == df["home"]].copy()
    away_df = df[df["team"] == df["away"]].copy()

    # Save targets before leakage removal
    targets = home_df[["gameid", "pts"]].rename(
        columns={"pts": "home_pts"}
    )

    targets = targets.merge(
        away_df[["gameid", "pts"]].rename(
            columns={"pts": "away_pts"}
        ),
        on="gameid",
        how="inner",
    )

    targets["point_spread"] = (
        targets["home_pts"] - targets["away_pts"]
    )

    targets["total_points"] = (
        targets["home_pts"] + targets["away_pts"]
    )

    # Remove leakage and keep only pre-game-safe features
    home_df = remove_leakage_cols(home_df)
    away_df = remove_leakage_cols(away_df)

    home_df = keep_numeric_pregame_features(home_df)
    away_df = keep_numeric_pregame_features(away_df)

    home_df = home_df.add_prefix("home_")
    away_df = away_df.add_prefix("away_")

    matchups = home_df.merge(
        away_df,
        left_on="home_gameid",
        right_on="away_gameid",
        how="inner",
    )

    matchups = matchups.rename(columns={"home_gameid": "gameid"})
    matchups = matchups.drop(columns=["away_gameid"], errors="ignore")

    # Restore date/season
    if "home_date" in matchups.columns:
        matchups = matchups.rename(columns={"home_date": "date"})

    if "home_season" in matchups.columns:
        matchups = matchups.rename(columns={"home_season": "season"})

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

    # Add matchup interaction features
    def add_diff(home_col, away_col, output_col):
        if home_col in matchups.columns and away_col in matchups.columns:
            matchups[output_col] = matchups[home_col] - matchups[away_col]

    add_diff(
        "home_net_rating_roll_mean_5",
        "away_net_rating_roll_mean_5",
        "net_rating_diff",
    )

    add_diff(
        "home_off_rating_roll_mean_5",
        "away_def_rating_roll_mean_5",
        "home_off_vs_away_def",
    )

    add_diff(
        "away_off_rating_roll_mean_5",
        "home_def_rating_roll_mean_5",
        "away_off_vs_home_def",
    )

    add_diff(
        "home_possessions_roll_mean_5",
        "away_possessions_roll_mean_5",
        "pace_diff",
    )

    if {
        "home_possessions_roll_mean_5",
        "away_possessions_roll_mean_5",
    }.issubset(matchups.columns):
        matchups["expected_pace"] = (
            matchups["home_possessions_roll_mean_5"]
            + matchups["away_possessions_roll_mean_5"]
        ) / 2

    # Add targets back at the end
    matchups = matchups.merge(targets, on="gameid", how="inner")

    matchups = matchups.replace([np.inf, -np.inf], np.nan)

    if "date" in matchups.columns:
        matchups = matchups.sort_values("date").reset_index(drop=True)

    return matchups


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--team-master",
        required=True,
        help="Path to team_master.csv or team_master.parquet",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to save score_matchups.parquet or .csv",
    )

    args = parser.parse_args()

    team_master = load_data(args.team_master)
    print(f"Loaded: {team_master.shape}")

    team_features = build_team_features(team_master)
    print(f"Features built: {team_features.shape}")

    score_matchups = build_matchups(team_features)
    print(f"Matchups built: {score_matchups.shape}")

    # Safety check for obvious leakage
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

    if suspicious_cols:
        print("\nWARNING: Possible leakage columns found:")
        for col in suspicious_cols:
            print("-", col)
    else:
        print("Leakage safety check passed.")

    save_data(score_matchups, args.output)
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()