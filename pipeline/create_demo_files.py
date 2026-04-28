# Path lets us build file paths in a clean, OS-independent way
from pathlib import Path

# pandas is used to load the full processed datasets and save the slimmed-down demo versions
import pandas as pd


# the folder where both demo CSV files will be saved
# the deployed Streamlit app reads from this folder because the full datasets are too large to host
OUT_DIR = Path("data/demo")

# create the folder if it doesn't already exist — parents=True handles any missing parent folders
OUT_DIR.mkdir(parents=True, exist_ok=True)


# creates a lightweight version of the win matchup dataset for the hosted demo app
# the full win_matchups dataset is very large, so we trim it down to the most recent 3000 games
# and only keep the columns the win predictor page actually needs
def create_win_demo():
    # load the full processed win matchup dataset
    df = pd.read_parquet("data/processed/win_matchups.parquet")

    # decide which columns to keep in the demo file:
    # - identifier and metadata columns (gameid, date, season, home, away, home_win)
    # - any rolling feature column (column names containing "_roll_")
    # - any matchup difference column (column names starting with "diff_")
    # everything else (raw box scores, advanced metrics, etc.) is left out to keep the file small
    keep_cols = [
        c for c in df.columns
        if c in ["gameid", "date", "season", "home", "away", "home_win"]
        or "_roll_" in c
        or c.startswith("diff_")
    ]

    # take only the most recent 3000 game rows — these are the freshest data points
    # using .tail() guarantees we get the end of the chronologically sorted dataset
    df = df[keep_cols].tail(3000).copy()

    # save as a CSV so the demo app can load it quickly without needing parquet support
    df.to_csv(OUT_DIR / "win_demo.csv", index=False)
    print("Saved data/demo/win_demo.csv", df.shape)


# creates a lightweight version of the player stats dataset for the hosted demo app
# the full player dataset has hundreds of thousands of rows, so we trim it to the most recent 20000
# and only keep the columns the player predictor page actually needs
def create_player_demo():
    # load the full processed player stats dataset
    df = pd.read_parquet("data/processed/player_stats_dataset.parquet")

    # decide which columns to keep in the demo file:
    # - key identifier and context columns (gameid, date, player, team, position, min)
    # - the actual stat columns used as targets (pts, reb, ast) so the history table can show them
    # - any rolling feature column — these are what the model uses to make predictions
    keep_cols = [
        c for c in df.columns
        if c in ["gameid", "date", "player", "team", "position", "min", "pts", "reb", "ast"]
        or "_roll_" in c
    ]

    # take the most recent 20000 player-game rows
    # 20000 is large enough to include a good spread of players while staying deployable
    df = df[keep_cols].tail(20000).copy()

    # save as a CSV for the demo app
    df.to_csv(OUT_DIR / "player_demo.csv", index=False)
    print("Saved data/demo/player_demo.csv", df.shape)


# run both demo file creation functions when this script is executed directly
# run this script whenever the full processed datasets are updated so the demo files stay fresh
if __name__ == "__main__":
    create_win_demo()
    create_player_demo()
