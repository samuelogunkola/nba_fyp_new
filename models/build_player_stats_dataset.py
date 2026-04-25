import pandas as pd
import numpy as np


# =========================
# LOAD DATA
# =========================
df = pd.read_parquet("data/processed/player_features.parquet")

print("Loaded player_features shape:", df.shape)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["date", "gameid", "playerid"]).reset_index(drop=True)


# =========================
# VALIDATION
# =========================
required_cols = [
    "gameid", "date", "playerid", "player",
    "team", "home", "away", "pts", "ast", "reb", "min"
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")


# =========================
# KEEP PRE-GAME SAFE FEATURES
# =========================
safe_exact_cols = [
    "gameid", "date", "playerid", "player",
    "team", "home", "away",
    "pts", "ast", "reb", "min", "position"
]

safe_contains = [
    "_rolling_", "_roll_", "_lag_", "_expanding_", "_hist_", "_prev_"
]

def is_safe(col):
    if col in safe_exact_cols:
        return True
    if any(x in col for x in safe_contains):
        return True
    return False


safe_cols = [c for c in df.columns if is_safe(c)]
player_df = df[safe_cols].copy()

print("Pregame-safe columns retained:", len(player_df.columns))


# =========================
# REMOVE LOW-MINUTE GAMES
# =========================
before = len(player_df)

player_df = player_df[player_df["min"].fillna(0) >= 5].copy()

after = len(player_df)

print("Removed low-minute rows:", before - after)


# =========================
# REMOVE LOW-SAMPLE PLAYERS
# =========================
counts = player_df.groupby("playerid").size()
valid_players = counts[counts >= 10].index

before = len(player_df)
player_df = player_df[player_df["playerid"].isin(valid_players)].copy()
after = len(player_df)

print("Removed low-sample players:", before - after)


# =========================
# FINAL CLEANUP
# =========================
player_df = player_df.sort_values(["date", "gameid", "playerid"]).reset_index(drop=True)

output_path = "data/processed/player_stats_dataset.parquet"
player_df.to_parquet(output_path, index=False)

print("\nSaved to:", output_path)
print("Final dataset shape:", player_df.shape)

print("\nTarget summary:")
print(player_df[["pts", "ast", "reb", "min"]].describe())