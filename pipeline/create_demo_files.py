from pathlib import Path
import pandas as pd

OUT_DIR = Path("data/demo")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def create_win_demo():
    df = pd.read_parquet("data/processed/win_matchups.parquet")

    keep_cols = [
        c for c in df.columns
        if c in ["gameid", "date", "season", "home", "away", "home_win"]
        or "_roll_" in c
        or c.startswith("diff_")
    ]

    df = df[keep_cols].tail(3000).copy()
    df.to_csv(OUT_DIR / "win_demo.csv", index=False)
    print("Saved data/demo/win_demo.csv", df.shape)


def create_player_demo():
    df = pd.read_parquet("data/processed/player_stats_dataset.parquet")

    keep_cols = [
        c for c in df.columns
        if c in ["gameid", "date", "player", "team", "position", "min", "pts", "reb", "ast"]
        or "_roll_" in c
    ]

    df = df[keep_cols].tail(20000).copy()
    df.to_csv(OUT_DIR / "player_demo.csv", index=False)
    print("Saved data/demo/player_demo.csv", df.shape)


if __name__ == "__main__":
    create_win_demo()
    create_player_demo()