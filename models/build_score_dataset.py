import pandas as pd
import numpy as np


# =========================
# LOAD DATA
# =========================
df = pd.read_parquet("data/processed/team_features.parquet")

print("Loaded team_features shape:", df.shape)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["date", "gameid", "team"]).reset_index(drop=True)


# =========================
# VALIDATION
# =========================
required_cols = ["gameid", "date", "team", "home", "away", "pts", "season"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")


# =========================
# PRE-GAME SAFE FEATURES + TARGETS
# =========================
safe_exact_cols = [
    "gameid",
    "date",
    "team",
    "home",
    "away",
    "season",
    "pts",        # target source
    "rest_days",
]

safe_contains = [
    "_rolling_",
    "_roll_",
    "_lag_",
    "_expanding_",
    "_hist_",
    "_prev_",
    "rest_days",
]

def is_safe_feature(col: str) -> bool:
    if col in safe_exact_cols:
        return True
    if any(x in col for x in safe_contains):
        return True
    return False


safe_cols = [c for c in df.columns if is_safe_feature(c)]
safe_df = df[safe_cols].copy()

print("Pregame-safe columns retained:", len(safe_df.columns))


# =========================
# ENSURE 2 TEAM ROWS PER GAME
# =========================
game_counts = safe_df.groupby("gameid").size()
valid_games = game_counts[game_counts == 2].index

safe_df = safe_df[safe_df["gameid"].isin(valid_games)].copy()

print("Games retained:", len(valid_games))


# =========================
# SPLIT HOME / AWAY
# =========================
home_df = safe_df[safe_df["team"] == safe_df["home"]].copy()
away_df = safe_df[safe_df["team"] == safe_df["away"]].copy()

print("Home rows:", home_df.shape)
print("Away rows:", away_df.shape)


# =========================
# RENAME COLUMNS
# =========================
id_cols = ["gameid", "date", "season", "home", "away"]

home_df = home_df.rename(columns={
    c: f"home_{c}" for c in home_df.columns if c not in id_cols
})

away_df = away_df.rename(columns={
    c: f"away_{c}" for c in away_df.columns if c not in id_cols
})


# =========================
# MERGE INTO MATCHUPS
# =========================
matchups = home_df.merge(
    away_df,
    on=id_cols,
    how="inner",
    validate="one_to_one"
)

print("Matchup dataset shape before diffs:", matchups.shape)


# =========================
# TARGETS
# =========================
if "home_pts" not in matchups.columns:
    raise ValueError("Expected column 'home_pts' not found after merge.")
if "away_pts" not in matchups.columns:
    raise ValueError("Expected column 'away_pts' not found after merge.")

matchups["home_pts"] = pd.to_numeric(matchups["home_pts"], errors="raise")
matchups["away_pts"] = pd.to_numeric(matchups["away_pts"], errors="raise")


# =========================
# CREATE DIFFERENCE FEATURES
# =========================
exclude_for_diff = {"gameid", "date", "season", "home", "away", "home_pts", "away_pts"}

home_feature_cols = [
    c for c in matchups.columns
    if c.startswith("home_") and c not in exclude_for_diff
]

diff_data = {}
created_diff_count = 0
skipped_non_numeric = 0

for home_col in home_feature_cols:
    away_col = home_col.replace("home_", "away_", 1)

    if away_col in matchups.columns:
        if pd.api.types.is_numeric_dtype(matchups[home_col]) and pd.api.types.is_numeric_dtype(matchups[away_col]):
            diff_col = home_col.replace("home_", "diff_", 1)
            diff_data[diff_col] = matchups[home_col] - matchups[away_col]
            created_diff_count += 1
        else:
            skipped_non_numeric += 1

if diff_data:
    diff_df = pd.DataFrame(diff_data, index=matchups.index)
    matchups = pd.concat([matchups, diff_df], axis=1)

print("Difference features created:", created_diff_count)
print("Non-numeric pairs skipped:", skipped_non_numeric)


# =========================
# FINAL FORMAT
# =========================
base_cols = ["gameid", "date", "season", "home", "away", "home_pts", "away_pts"]
other_cols = [c for c in matchups.columns if c not in base_cols]

matchups = matchups[base_cols + other_cols]
matchups = matchups.sort_values("date").reset_index(drop=True)


# =========================
# SAVE
# =========================
output_path = "data/processed/score_matchups.parquet"
matchups.to_parquet(output_path, index=False)

print("\nSaved to:", output_path)
print("Final dataset shape:", matchups.shape)
print("\nTarget summary:")
print(matchups[["home_pts", "away_pts"]].describe())