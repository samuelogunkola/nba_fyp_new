from pathlib import Path
import pandas as pd


TEAM_MASTER = Path("data/processed/team_master.csv")
SCORE_DETAILS = Path("models/experiments/score/artifacts/score_prediction_details.csv")
OUTPUT = Path("models/experiments/score/artifacts/score_prediction_details.csv")


def main():
    team_df = pd.read_csv(TEAM_MASTER)
    score_df = pd.read_csv(SCORE_DETAILS)

    required = {"gameid", "team", "home", "away"}
    missing = required - set(team_df.columns)

    if missing:
        raise ValueError(f"team_master missing columns: {missing}")

    game_teams = (
        team_df[["gameid", "home", "away"]]
        .drop_duplicates(subset=["gameid"])
        .rename(columns={"home": "home_team", "away": "away_team"})
    )

    merged = score_df.merge(game_teams, on="gameid", how="left")

    # Reorder columns
    front_cols = [
        "gameid",
        "date",
        "home_team",
        "away_team",
        "home_pts",
        "away_pts",
        "pred_home_pts",
        "pred_away_pts",
    ]

    front_cols = [c for c in front_cols if c in merged.columns]
    other_cols = [c for c in merged.columns if c not in front_cols]

    merged = merged[front_cols + other_cols]

    merged.to_csv(OUTPUT, index=False)

    print("Patched score prediction file with team names.")
    print("Saved to:", OUTPUT)
    print("Columns:", merged.columns.tolist())


if __name__ == "__main__":
    main()