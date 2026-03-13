#!/usr/bin/env python3
"""
Merge UEFA Champions League (CL) + Europa League (EL) matches from openfootball
into existing Soccer/soccer_data/matches.csv.

Source:
- https://github.com/openfootball/champions-league
  season folders with cl.txt / el.txt

Usage:
  python Soccer\\soccer_merge_uefa.py --data-dir Soccer\\soccer_data --start-season 2011 --end-season 2025
"""

import argparse
import json
import os
import re
from datetime import datetime, timezone

import pandas as pd
import requests

BASE_RAW = "https://raw.githubusercontent.com/openfootball/champions-league/master"

MONTH_MAP = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def season_dir(start_year: int) -> str:
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def clean_team_name(name: str) -> str:
    s = str(name).strip()
    # remove trailing country code marker e.g. "(ENG)"
    s = re.sub(r"\s+\([A-Z]{2,3}\)$", "", s).strip()
    return s


def parse_date_line(line: str):
    # Examples:
    # "  Wed Sep/25 2024"
    # "  Thu May/08"
    txt = line.strip()
    m = re.search(r"([A-Za-z]{3})/(\d{1,2})(?:\s+(\d{4}))?", txt)
    if not m:
        return None
    mon = MONTH_MAP.get(m.group(1).lower())
    day = int(m.group(2))
    year = int(m.group(3)) if m.group(3) else None
    if mon is None:
        return None
    return mon, day, year


def parse_match_line(line: str):
    # Match lines usually look like:
    # "21.00  Team A (...) v Team B (...)  2-1 (1-0)"
    # or without kickoff time at start.
    txt = line.rstrip()
    m = re.match(r"^\s*(?:(\d{1,2}\.\d{2})\s+)?(.+?)\s+v\s+(.+?)\s+(\d+)-(\d+)(?:\s+\(.*\))?\s*$", txt)
    if not m:
        return None
    home = clean_team_name(m.group(2))
    away = clean_team_name(m.group(3))
    hg = int(m.group(4))
    ag = int(m.group(5))
    return home, away, hg, ag


def parse_openfootball_txt(text: str, comp_code: str, comp_name: str, season_start: int):
    rows = []
    current_year = None
    current_date = None

    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            continue

        maybe_date = parse_date_line(line)
        if maybe_date:
            mon, day, year = maybe_date
            if year is not None:
                current_year = year
            if current_year is None:
                # fallback by season window
                current_year = season_start if mon >= 7 else season_start + 1
            try:
                current_date = datetime(current_year, mon, day)
            except ValueError:
                current_date = None
            continue

        pm = parse_match_line(line)
        if pm and current_date is not None:
            home, away, hg, ag = pm
            rows.append(
                {
                    "competition_code": comp_code,
                    "competition_name": comp_name,
                    "season_start": season_start,
                    "season_label": f"{season_start}-{(season_start + 1) % 100:02d}",
                    "match_date": pd.Timestamp(current_date.date()),
                    "home_team": home,
                    "away_team": away,
                    "home_goals": float(hg),
                    "away_goals": float(ag),
                    "result": "H" if hg > ag else ("A" if hg < ag else "D"),
                    "b365_home_odds": pd.NA,
                    "b365_draw_odds": pd.NA,
                    "b365_away_odds": pd.NA,
                }
            )

    return pd.DataFrame(rows)


def fetch_openfootball_comp(season_start: int, filename: str, comp_code: str, comp_name: str):
    sd = season_dir(season_start)
    url = f"{BASE_RAW}/{sd}/{filename}"
    try:
        r = requests.get(url, timeout=40)
        if r.status_code != 200:
            return pd.DataFrame(), url, r.status_code
        df = parse_openfootball_txt(r.text, comp_code, comp_name, season_start)
        return df, url, 200
    except Exception:
        return pd.DataFrame(), url, -1


def main():
    parser = argparse.ArgumentParser(description="Merge CL/EL from openfootball into matches.csv")
    parser.add_argument("--data-dir", default="Soccer\\soccer_data")
    parser.add_argument("--start-season", type=int, default=2011)
    parser.add_argument("--end-season", type=int, default=2025)
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    matches_path = os.path.join(args.data_dir, "matches.csv")
    if not os.path.exists(matches_path):
        raise FileNotFoundError(f"Missing base matches file: {matches_path}")

    base = pd.read_csv(matches_path)
    for c in [
        "competition_code",
        "competition_name",
        "season_start",
        "season_label",
        "match_date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "result",
        "b365_home_odds",
        "b365_draw_odds",
        "b365_away_odds",
    ]:
        if c not in base.columns:
            base[c] = pd.NA

    base["match_date"] = pd.to_datetime(base["match_date"], errors="coerce")

    logs = []
    add_frames = []

    for season in range(args.start_season, args.end_season + 1):
        # CL
        cl_df, cl_url, cl_status = fetch_openfootball_comp(season, "cl.txt", "CL", "UEFA Champions League")
        add_frames.append(cl_df)
        logs.append(
            {
                "season_start": season,
                "competition_code": "CL",
                "url": cl_url,
                "http_status": cl_status,
                "rows": int(len(cl_df)),
            }
        )

        # EL
        el_df, el_url, el_status = fetch_openfootball_comp(season, "el.txt", "EL", "UEFA Europa League")
        add_frames.append(el_df)
        logs.append(
            {
                "season_start": season,
                "competition_code": "EL",
                "url": el_url,
                "http_status": el_status,
                "rows": int(len(el_df)),
            }
        )

    add = pd.concat(add_frames, ignore_index=True) if add_frames else pd.DataFrame()
    if len(add) > 0:
        add["match_date"] = pd.to_datetime(add["match_date"], errors="coerce")

    pre_rows = len(base)

    merged = pd.concat([base, add], ignore_index=True)
    merged = merged.dropna(subset=["match_date", "home_team", "away_team", "home_goals", "away_goals"])

    # Deduplicate by core match identity.
    merged["_k_date"] = merged["match_date"].dt.strftime("%Y-%m-%d")
    merged["_k_home"] = merged["home_team"].astype(str).str.strip().str.lower()
    merged["_k_away"] = merged["away_team"].astype(str).str.strip().str.lower()
    merged["_k_comp"] = merged["competition_code"].astype(str).str.strip().str.upper()
    merged = merged.sort_values(["_k_date", "_k_comp", "_k_home", "_k_away"]).drop_duplicates(
        subset=["_k_date", "_k_comp", "_k_home", "_k_away"], keep="first"
    )
    merged = merged.drop(columns=["_k_date", "_k_home", "_k_away", "_k_comp"])

    merged = merged.sort_values(["match_date", "competition_code", "home_team", "away_team"]).reset_index(drop=True)

    post_rows = len(merged)
    added_net = post_rows - pre_rows

    merged.to_csv(matches_path, index=False)

    log_df = pd.DataFrame(logs)
    log_path = os.path.join(args.data_dir, "uefa_merge_log.csv")
    log_df.to_csv(log_path, index=False)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_rows_before": int(pre_rows),
        "rows_fetched_cl_el_raw": int(len(add)),
        "base_rows_after": int(post_rows),
        "net_rows_added": int(added_net),
        "cl_rows_added": int((add["competition_code"] == "CL").sum()) if len(add) else 0,
        "el_rows_added": int((add["competition_code"] == "EL").sum()) if len(add) else 0,
        "log_file": log_path,
    }

    summary_path = os.path.join(args.data_dir, "uefa_merge_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[DONE] UEFA CL/EL merge complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
