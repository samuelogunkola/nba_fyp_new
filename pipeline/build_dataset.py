from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
TRAD_DIR = BASE_DIR / "data" / "raw" / "nba_traditional"
REF_DIR = BASE_DIR / "data" / "raw" / "nba_reference"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace("%", "pct", regex=False)
        .str.replace("+/-", "plus_minus", regex=False)
    )
    return df


def convert_minutes(mp: object) -> float:
    if pd.isna(mp):
        return 0.0
    try:
        return float(mp)
    except (ValueError, TypeError):
        return 0.0


def build_player_master() -> pd.DataFrame:
    player_path = TRAD_DIR / "traditional.csv"
    player_info_path = REF_DIR / "player_info.csv"

    players = pd.read_csv(player_path, low_memory=False)
    player_info = pd.read_csv(player_info_path, low_memory=False)

    players = clean_columns(players)
    player_info = clean_columns(player_info)

    player_info = player_info.rename(
        columns={
            "playername": "player",
            "from": "career_start_year",
            "to": "career_end_year",
            "pos": "position",
            "ht": "height",
            "wt": "weight",
            "birthdate": "birth_date",
            "colleges": "college",
        }
    )

    players["date"] = pd.to_datetime(players["date"], errors="coerce")
    players["min"] = players["min"].apply(convert_minutes)

    player_master = players.merge(player_info, on="player", how="left")
    player_master = player_master.drop_duplicates()

    return player_master


def build_team_master() -> pd.DataFrame:
    team_path = TRAD_DIR / "team_traditional.csv"

    teams = pd.read_csv(team_path, low_memory=False)
    teams = clean_columns(teams)

    teams["date"] = pd.to_datetime(teams["date"], errors="coerce")
    teams["min"] = teams["min"].apply(convert_minutes)
    teams = teams.drop_duplicates()

    return teams


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    player_master = build_player_master()
    team_master = build_team_master()

    player_out = PROCESSED_DIR / "player_master.csv"
    team_out = PROCESSED_DIR / "team_master.csv"

    player_master.to_csv(player_out, index=False)
    team_master.to_csv(team_out, index=False)

    print("Processed datasets created successfully.")
    print("player_master shape:", player_master.shape)
    print("team_master shape:", team_master.shape)
    print("Saved to:", player_out)
    print("Saved to:", team_out)


if __name__ == "__main__":
    main()