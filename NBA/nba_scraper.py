#!/usr/bin/env python3
"""
NBA historical data scraper (NBA only, league id 00).

Outputs:
- teams.csv
- franchise_history.csv
- raw_team_game_logs.csv
- games.csv
- metadata.json

Usage:
  python nba_scraper.py --start-season 2000 --end-season 2025 --outdir nba_data
"""

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog, teamyearbyyearstats
from nba_api.stats.library.http import NBAStatsHTTP
from nba_api.stats.static import teams as static_teams


NBA_LEAGUE_ID = "00"
SEASON_TYPES = ["Regular Season", "Playoffs"]


# Browser-like headers improve API reliability.
NBAStatsHTTP.STATS_HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Connection": "keep-alive",
}


@dataclass
class RetryConfig:
    max_attempts: int = 6
    base_sleep_sec: float = 0.8
    jitter_sec: float = 0.5


def season_str(start_year: int) -> str:
    """Convert 2024 -> '2024-25'."""
    end_two = (start_year + 1) % 100
    return f"{start_year}-{end_two:02d}"


def call_with_retry(fn, retry: RetryConfig, label: str):
    last_err = None
    for attempt in range(1, retry.max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # nba_api raises endpoint-specific exceptions
            last_err = exc
            sleep_for = retry.base_sleep_sec * (2 ** (attempt - 1)) + random.uniform(0.0, retry.jitter_sec)
            print(f"[WARN] {label} failed (attempt {attempt}/{retry.max_attempts}): {exc}")
            if attempt < retry.max_attempts:
                time.sleep(sleep_for)
    raise RuntimeError(f"Failed after retries: {label}") from last_err


def fetch_teams() -> pd.DataFrame:
    rows = static_teams.get_teams()
    df = pd.DataFrame(rows).rename(
        columns={
            "id": "team_id",
            "full_name": "team_full_name",
            "abbreviation": "team_abbreviation",
            "nickname": "team_nickname",
            "city": "team_city",
            "state": "team_state",
            "year_founded": "year_founded",
        }
    )
    return df.sort_values(["year_founded", "team_full_name"]).reset_index(drop=True)


def fetch_franchise_history(team_ids: List[int], retry: RetryConfig) -> pd.DataFrame:
    out = []
    for tid in team_ids:
        label = f"TeamYearByYearStats team_id={tid}"
        resp = call_with_retry(
            lambda: teamyearbyyearstats.TeamYearByYearStats(team_id=tid),
            retry=retry,
            label=label,
        )
        df = resp.get_data_frames()[0]
        if len(df) == 0:
            continue
        out.append(df)
        # Light pacing avoids API throttling.
        time.sleep(0.25 + random.uniform(0.0, 0.2))

    if not out:
        return pd.DataFrame()

    hist = pd.concat(out, ignore_index=True)
    hist = hist.rename(
        columns={
            "TEAM_ID": "team_id",
            "TEAM_CITY": "team_city",
            "TEAM_NAME": "team_name",
            "YEAR": "season_label",
            "WINS": "wins",
            "LOSSES": "losses",
            "WIN_PCT": "win_pct",
            "CONF_RANK": "conf_rank",
            "DIV_RANK": "div_rank",
            "PO_WINS": "playoff_wins",
            "PO_LOSSES": "playoff_losses",
            "NBA_FINALS_APPEARANCE": "nba_finals_appearance",
            "LEAGUE_ID": "league_id",
        }
    )
    if "league_id" in hist.columns:
        hist = hist[hist["league_id"].astype(str) == NBA_LEAGUE_ID].copy()
    return hist.sort_values(["team_id", "season_label"]).reset_index(drop=True)


def fetch_team_game_logs(start_season: int, end_season: int, retry: RetryConfig) -> pd.DataFrame:
    frames = []
    for year in range(start_season, end_season + 1):
        season = season_str(year)
        for season_type in SEASON_TYPES:
            label = f"LeagueGameLog {season} {season_type}"
            resp = call_with_retry(
                lambda s=season, st=season_type: leaguegamelog.LeagueGameLog(
                    season=s,
                    season_type_all_star=st,
                    player_or_team_abbreviation="T",
                    league_id=NBA_LEAGUE_ID,
                ),
                retry=retry,
                label=label,
            )
            df = resp.get_data_frames()[0]
            if len(df) == 0:
                continue

            df["SEASON"] = season
            df["SEASON_TYPE"] = season_type
            frames.append(df)
            print(f"[OK] {season} | {season_type:<14} -> {len(df):,} team-game rows")
            time.sleep(0.45 + random.uniform(0.0, 0.25))

    if not frames:
        return pd.DataFrame()

    logs = pd.concat(frames, ignore_index=True)
    logs = logs.rename(
        columns={
            "SEASON_ID": "season_id",
            "TEAM_ID": "team_id",
            "TEAM_ABBREVIATION": "team_abbreviation",
            "TEAM_NAME": "team_name",
            "GAME_ID": "game_id",
            "GAME_DATE": "game_date",
            "MATCHUP": "matchup",
            "WL": "wl",
            "MIN": "minutes",
            "FGM": "fgm",
            "FGA": "fga",
            "FG_PCT": "fg_pct",
            "FG3M": "fg3m",
            "FG3A": "fg3a",
            "FG3_PCT": "fg3_pct",
            "FTM": "ftm",
            "FTA": "fta",
            "FT_PCT": "ft_pct",
            "OREB": "oreb",
            "DREB": "dreb",
            "REB": "reb",
            "AST": "ast",
            "STL": "stl",
            "BLK": "blk",
            "TOV": "tov",
            "PF": "pf",
            "PTS": "pts",
            "PLUS_MINUS": "plus_minus",
            "VIDEO_AVAILABLE": "video_available",
            "SEASON": "season",
            "SEASON_TYPE": "season_type",
        }
    )
    logs["game_date"] = pd.to_datetime(logs["game_date"], errors="coerce")
    logs = logs.sort_values(["game_date", "game_id", "team_id"]).reset_index(drop=True)
    return logs


def parse_home_away(matchup: str) -> Tuple[bool, str]:
    """Return (is_home, opponent_abbreviation)."""
    text = str(matchup)
    if " vs. " in text:
        left, right = text.split(" vs. ")
        return True, right.strip()
    if " @ " in text:
        left, right = text.split(" @ ")
        return False, right.strip()
    return False, ""


def build_games_table(team_logs: pd.DataFrame) -> pd.DataFrame:
    if len(team_logs) == 0:
        return pd.DataFrame()

    logs = team_logs.copy()
    parsed = logs["matchup"].apply(parse_home_away)
    logs["is_home"] = parsed.apply(lambda x: x[0])
    logs["opponent_abbreviation"] = parsed.apply(lambda x: x[1])

    rows = []
    for game_id, g in logs.groupby("game_id", sort=False):
        if len(g) != 2:
            # Extremely rare data integrity issue; skip malformed pairs.
            continue
        g = g.sort_values("is_home")  # away first, home second
        away = g.iloc[0]
        home = g.iloc[1]

        rows.append(
            {
                "game_id": game_id,
                "game_date": home["game_date"],
                "season": home["season"],
                "season_type": home["season_type"],
                "home_team_id": int(home["team_id"]),
                "home_team_abbreviation": home["team_abbreviation"],
                "home_team_name": home["team_name"],
                "away_team_id": int(away["team_id"]),
                "away_team_abbreviation": away["team_abbreviation"],
                "away_team_name": away["team_name"],
                "home_pts": float(home["pts"]),
                "away_pts": float(away["pts"]),
                "home_win": int(home["wl"] == "W"),
                "point_diff_home_minus_away": float(home["pts"] - away["pts"]),
                "overtime_flag": int(float(home["minutes"]) > 240.0),
            }
        )

    games = pd.DataFrame(rows)
    games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    return games


def main():
    parser = argparse.ArgumentParser(description="NBA-only historical data scraper")
    parser.add_argument("--start-season", type=int, default=2000, help="Start season year, e.g. 2000 => 2000-01")
    parser.add_argument("--end-season", type=int, default=2025, help="End season year, e.g. 2025 => 2025-26")
    parser.add_argument("--outdir", type=str, default="nba_data", help="Output directory")
    parser.add_argument("--max-attempts", type=int, default=6, help="Retries per API call")
    parser.add_argument("--base-sleep", type=float, default=0.8, help="Base sleep for retry backoff")
    args = parser.parse_args()

    if args.end_season < args.start_season:
        raise ValueError("--end-season must be >= --start-season")

    os.makedirs(args.outdir, exist_ok=True)
    retry = RetryConfig(max_attempts=args.max_attempts, base_sleep_sec=args.base_sleep)

    print("=" * 72)
    print("NBA DATA SCRAPER (NBA ONLY)")
    print("=" * 72)
    print(f"[CONFIG] seasons: {args.start_season}-{args.end_season}")
    print(f"[CONFIG] output : {os.path.abspath(args.outdir)}")

    teams = fetch_teams()
    teams.to_csv(os.path.join(args.outdir, "teams.csv"), index=False)
    print(f"[SAVE] teams.csv -> {len(teams):,} teams")

    franchise_hist = fetch_franchise_history(teams["team_id"].tolist(), retry=retry)
    franchise_hist.to_csv(os.path.join(args.outdir, "franchise_history.csv"), index=False)
    print(f"[SAVE] franchise_history.csv -> {len(franchise_hist):,} rows")

    team_logs = fetch_team_game_logs(args.start_season, args.end_season, retry=retry)
    team_logs.to_csv(os.path.join(args.outdir, "raw_team_game_logs.csv"), index=False)
    print(f"[SAVE] raw_team_game_logs.csv -> {len(team_logs):,} team-game rows")

    games = build_games_table(team_logs)
    games.to_csv(os.path.join(args.outdir, "games.csv"), index=False)
    print(f"[SAVE] games.csv -> {len(games):,} games")

    metadata = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "league": "NBA",
        "league_id": NBA_LEAGUE_ID,
        "seasons": {
            "start": args.start_season,
            "end": args.end_season,
            "labels": [season_str(y) for y in range(args.start_season, args.end_season + 1)],
        },
        "season_types": SEASON_TYPES,
        "row_counts": {
            "teams": int(len(teams)),
            "franchise_history": int(len(franchise_hist)),
            "raw_team_game_logs": int(len(team_logs)),
            "games": int(len(games)),
        },
        "files": [
            "teams.csv",
            "franchise_history.csv",
            "raw_team_game_logs.csv",
            "games.csv",
        ],
    }

    with open(os.path.join(args.outdir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print("[SAVE] metadata.json")

    print("=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()

