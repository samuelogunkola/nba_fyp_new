import pandas as pd

games = pd.read_csv(r"C:\Users\samue\OneDrive\Documents\FYP_NBATrackerPredictor\nba_fyp\data\raw\nba_reference\games.csv", low_memory=False)
boxscore = pd.read_csv(r"C:\Users\samue\OneDrive\Documents\FYP_NBATrackerPredictor\nba_fyp\data\raw\nba_reference\boxscore.csv", low_memory=False)
players = pd.read_csv(r"C:\Users\samue\OneDrive\Documents\FYP_NBATrackerPredictor\nba_fyp\data\raw\nba_reference\player_info.csv", low_memory=False)

print("GAMES COLUMNS:")
print(games.columns)

print("\nBOXSCORE COLUMNS:")
print(boxscore.columns)

print("\nPLAYER INFO COLUMNS:")
print(players.columns)

