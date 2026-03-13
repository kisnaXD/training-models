#!/usr/bin/env python3
"""
European Club Soccer scraper from football-data.co.uk historical CSVs.

Club-only scope:
- Major domestic leagues (Europe)
- UEFA Champions League / Europa League / Conference League

Usage:
  python Soccer\\soccer_scraper.py --start-season 2010 --end-season 2025 --outdir Soccer\\soccer_data
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd
import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281"

# European club competitions only.
COMPETITIONS = {
    # Domestic leagues
    "E0": "Premier League",
    "SP1": "La Liga",
    "I1": "Serie A",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
    "N1": "Eredivisie",
    "P1": "Primeira Liga",
    "B1": "Belgian Pro League",
    "SC0": "Scottish Premiership",
    "T1": "Turkish Super Lig",
    # UEFA club competitions
    "CL": "UEFA Champions League",
    "EL": "UEFA Europa League",
    "EC": "UEFA Conference League",
}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def season_token(start_year: int) -> str:
    # 2024 -> "2425"
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def fetch_csv(url: str, retries: int = 3) -> pd.DataFrame:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=40)
            if r.status_code == 404:
                return pd.DataFrame()
            r.raise_for_status()
            if not r.text.strip():
                return pd.DataFrame()
            return pd.read_csv(pd.io.common.StringIO(r.text), low_memory=False)
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(1.2 * attempt)
    print(f"[WARN] failed: {url} -> {last_err}")
    return pd.DataFrame()


def normalize_match_df(df: pd.DataFrame, comp_code: str, comp_name: str, season_start: int) -> pd.DataFrame:
    if len(df) == 0:
        return df

    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    date_col = pick("Date")
    home_col = pick("HomeTeam")
    away_col = pick("AwayTeam")
    hg_col = pick("FTHG")
    ag_col = pick("FTAG")
    res_col = pick("FTR")

    if not all([date_col, home_col, away_col, hg_col, ag_col, res_col]):
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "competition_code": comp_code,
            "competition_name": comp_name,
            "season_start": season_start,
            "season_label": f"{season_start}-{(season_start + 1) % 100:02d}",
            "match_date": pd.to_datetime(df[date_col], errors="coerce", dayfirst=True),
            "home_team": df[home_col].astype(str).str.strip(),
            "away_team": df[away_col].astype(str).str.strip(),
            "home_goals": pd.to_numeric(df[hg_col], errors="coerce"),
            "away_goals": pd.to_numeric(df[ag_col], errors="coerce"),
            "result": df[res_col].astype(str).str.strip(),
            "b365_home_odds": pd.to_numeric(df[pick("B365H")] if pick("B365H") else pd.Series([None] * len(df)), errors="coerce"),
            "b365_draw_odds": pd.to_numeric(df[pick("B365D")] if pick("B365D") else pd.Series([None] * len(df)), errors="coerce"),
            "b365_away_odds": pd.to_numeric(df[pick("B365A")] if pick("B365A") else pd.Series([None] * len(df)), errors="coerce"),
        }
    )

    out = out.dropna(subset=["match_date", "home_goals", "away_goals"])
    out = out[out["home_team"] != ""]
    out = out[out["away_team"] != ""]
    out = out.sort_values(["match_date", "home_team", "away_team"]).reset_index(drop=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="European club soccer scraper")
    parser.add_argument("--start-season", type=int, default=2010)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--outdir", type=str, default="Soccer\\soccer_data")
    args = parser.parse_args()

    if args.end_season < args.start_season:
        raise ValueError("--end-season must be >= --start-season")

    ensure_dir(args.outdir)

    all_rows: List[pd.DataFrame] = []
    log_rows: List[Dict[str, object]] = []

    seasons = list(range(args.start_season, args.end_season + 1))

    for season in seasons:
        tok = season_token(season)
        for code, name in COMPETITIONS.items():
            url = f"{BASE_URL}/{tok}/{code}.csv"
            raw = fetch_csv(url)
            norm = normalize_match_df(raw, code, name, season)
            status = "ok" if len(norm) > 0 else "missing_or_empty"
            if len(norm) > 0:
                all_rows.append(norm)
            log_rows.append(
                {
                    "season_start": season,
                    "competition_code": code,
                    "competition_name": name,
                    "url": url,
                    "raw_rows": int(len(raw)),
                    "normalized_rows": int(len(norm)),
                    "status": status,
                }
            )
            print(f"[{season}] {code:<3} -> {len(norm):,} rows ({status})")

    matches = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if len(matches) > 0:
        matches = matches.sort_values(["match_date", "competition_code", "home_team", "away_team"]).reset_index(drop=True)

    scrape_log = pd.DataFrame(log_rows)

    matches_path = os.path.join(args.outdir, "matches.csv")
    log_path = os.path.join(args.outdir, "scrape_log.csv")
    meta_path = os.path.join(args.outdir, "metadata.json")

    matches.to_csv(matches_path, index=False)
    scrape_log.to_csv(log_path, index=False)

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seasons": {"start": args.start_season, "end": args.end_season, "count": len(seasons)},
        "competitions": COMPETITIONS,
        "rows": {
            "matches": int(len(matches)),
            "scrape_log": int(len(scrape_log)),
            "successful_competition_season_pairs": int((scrape_log["normalized_rows"] > 0).sum()),
        },
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n[DONE] Soccer scrape complete")
    print(json.dumps(meta["rows"], indent=2))


if __name__ == "__main__":
    main()
