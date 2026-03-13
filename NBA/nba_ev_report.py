#!/usr/bin/env python3
"""
NBA EV report generator for listed matchups.

Builds current team state from historical raw logs, predicts matchup probabilities,
and compares model fair value vs market prices.
"""

import json
import os
from collections import defaultdict, deque
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

import nba_model_pipeline as mp


EDGE_THRESHOLD_CENTS = 3.0


def infer_season_label(dt: pd.Timestamp) -> str:
    # NBA season starts around Oct.
    if dt.month >= 10:
        start = dt.year
    else:
        start = dt.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


def build_state_snapshot(games: pd.DataFrame):
    elo = defaultdict(lambda: mp.ELO_BASE)
    team_hist = defaultdict(lambda: deque(maxlen=120))
    last_game_date = {}
    season_record = defaultdict(lambda: {"w": 0, "l": 0})
    current_season = None

    for _, g in games.iterrows():
        season = str(g["season"])
        if current_season is None:
            current_season = season
        if season != current_season:
            for tid in list(elo.keys()):
                elo[tid] = 0.75 * elo[tid] + 0.25 * mp.ELO_BASE
            season_record = defaultdict(lambda: {"w": 0, "l": 0})
            current_season = season

        home = int(g["home_team_id"])
        away = int(g["away_team_id"])
        home_win = int(g["home_pts"] > g["away_pts"])
        mov = float(g["home_pts"] - g["away_pts"])
        playoffs = str(g["season_type"]) == "Playoffs"

        home_elo = elo[home]
        away_elo = elo[away]

        home_efg = (g["home_fgm"] + 0.5 * g["home_fg3m"]) / max(g["home_fga"], 1.0)
        away_efg = (g["away_fgm"] + 0.5 * g["away_fg3m"]) / max(g["away_fga"], 1.0)

        team_hist[home].append(
            {
                "win": home_win,
                "pts": g["home_pts"],
                "opp_pts": g["away_pts"],
                "efg": home_efg,
                "tov": g["home_tov"],
                "reb": g["home_reb"],
                "ast": g["home_ast"],
                "opp_elo": away_elo,
            }
        )
        team_hist[away].append(
            {
                "win": 1 - home_win,
                "pts": g["away_pts"],
                "opp_pts": g["home_pts"],
                "efg": away_efg,
                "tov": g["away_tov"],
                "reb": g["away_reb"],
                "ast": g["away_ast"],
                "opp_elo": home_elo,
            }
        )

        season_record[(home, season)]["w"] += home_win
        season_record[(home, season)]["l"] += (1 - home_win)
        season_record[(away, season)]["w"] += (1 - home_win)
        season_record[(away, season)]["l"] += home_win
        last_game_date[home] = g["game_date"]
        last_game_date[away] = g["game_date"]

        nh, na = mp.elo_update(home_elo, away_elo, bool(home_win), mov, playoffs=playoffs)
        elo[home] = nh
        elo[away] = na

    return {
        "elo": elo,
        "team_hist": team_hist,
        "last_game_date": last_game_date,
        "season_record": season_record,
        "current_season": current_season,
    }


def make_feature_row(home_id, away_id, game_date, season_label, priors, state):
    season_start = mp.parse_season_start(season_label)

    elo = state["elo"]
    team_hist = state["team_hist"]
    last_game_date = state["last_game_date"]
    season_record = state["season_record"]

    home_elo = elo[home_id]
    away_elo = elo[away_id]

    home_rest = (game_date - last_game_date[home_id]).days if home_id in last_game_date else 5
    away_rest = (game_date - last_game_date[away_id]).days if away_id in last_game_date else 5

    h5 = mp.summarize_hist(team_hist[home_id], 5)
    h10 = mp.summarize_hist(team_hist[home_id], 10)
    h20 = mp.summarize_hist(team_hist[home_id], 20)
    a5 = mp.summarize_hist(team_hist[away_id], 5)
    a10 = mp.summarize_hist(team_hist[away_id], 10)
    a20 = mp.summarize_hist(team_hist[away_id], 20)

    h_rec = season_record[(home_id, season_label)]
    a_rec = season_record[(away_id, season_label)]
    h_games = h_rec["w"] + h_rec["l"]
    a_games = a_rec["w"] + a_rec["l"]
    h_wr = (h_rec["w"] / h_games) if h_games > 0 else 0.5
    a_wr = (a_rec["w"] / a_games) if a_games > 0 else 0.5

    hp = priors.get((home_id, season_start - 1), {"prior_win_pct": 0.5, "prior_playoff_wins": 0.0})
    ap = priors.get((away_id, season_start - 1), {"prior_win_pct": 0.5, "prior_playoff_wins": 0.0})

    return {
        "season_start": season_start,
        "home_team_id": home_id,
        "away_team_id": away_id,
        "playoffs": 0,
        "elo_diff": home_elo - away_elo,
        "home_elo": home_elo,
        "away_elo": away_elo,
        "rest_diff": home_rest - away_rest,
        "home_rest": home_rest,
        "away_rest": away_rest,
        "home_b2b": int(home_rest <= 1),
        "away_b2b": int(away_rest <= 1),
        "season_wr_diff": h_wr - a_wr,
        "home_season_wr": h_wr,
        "away_season_wr": a_wr,
        "home_season_games": h_games,
        "away_season_games": a_games,
        "prior_win_pct_diff": hp["prior_win_pct"] - ap["prior_win_pct"],
        "home_prior_win_pct": hp["prior_win_pct"],
        "away_prior_win_pct": ap["prior_win_pct"],
        "prior_playoff_wins_diff": hp["prior_playoff_wins"] - ap["prior_playoff_wins"],
        "wr5_diff": h5["win_rate"] - a5["win_rate"],
        "wr10_diff": h10["win_rate"] - a10["win_rate"],
        "wr20_diff": h20["win_rate"] - a20["win_rate"],
        "net5_diff": h5["net"] - a5["net"],
        "net10_diff": h10["net"] - a10["net"],
        "net20_diff": h20["net"] - a20["net"],
        "pts10_diff": h10["avg_pts"] - a10["avg_pts"],
        "opp_pts10_diff": h10["avg_opp_pts"] - a10["avg_opp_pts"],
        "efg10_diff": h10["avg_efg"] - a10["avg_efg"],
        "tov10_diff": h10["avg_tov"] - a10["avg_tov"],
        "reb10_diff": h10["avg_reb"] - a10["avg_reb"],
        "ast10_diff": h10["avg_ast"] - a10["avg_ast"],
        "opp_elo10_diff": h10["avg_opp_elo"] - a10["avg_opp_elo"],
        "home_games_hist": h20["count"],
        "away_games_hist": a20["count"],
    }


def symmetric_prob(team_a_id, team_b_id, game_date, season_label, priors, state, model, imputer, calibrator, feature_cols):
    # Neutralize unknown home/away by averaging both orientations.
    row_ab = make_feature_row(team_a_id, team_b_id, game_date, season_label, priors, state)
    row_ba = make_feature_row(team_b_id, team_a_id, game_date, season_label, priors, state)

    X_ab = pd.DataFrame([{c: row_ab.get(c, np.nan) for c in feature_cols}])[feature_cols]
    X_ba = pd.DataFrame([{c: row_ba.get(c, np.nan) for c in feature_cols}])[feature_cols]

    X_ab_i = imputer.transform(X_ab)
    X_ba_i = imputer.transform(X_ba)

    p_ab_raw = float(model.predict_proba(X_ab_i)[0, 1])
    p_ba_raw = float(model.predict_proba(X_ba_i)[0, 1])

    p_a_raw = 0.5 * (p_ab_raw + (1.0 - p_ba_raw))

    p_ab_cal = float(calibrator.transform([p_ab_raw])[0])
    p_ba_cal = float(calibrator.transform([p_ba_raw])[0])
    p_a_cal = 0.5 * (p_ab_cal + (1.0 - p_ba_cal))

    return p_a_raw, p_a_cal


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "nba_data")
    out_dir = os.path.join(base_dir, "nba_outputs")

    raw_logs = pd.read_csv(os.path.join(data_dir, "raw_team_game_logs.csv"))
    franchise = pd.read_csv(os.path.join(data_dir, "franchise_history.csv"))
    teams_df = pd.read_csv(os.path.join(data_dir, "teams.csv"))

    model = joblib.load(os.path.join(out_dir, "nba_xgb_model.pkl"))
    imputer = joblib.load(os.path.join(out_dir, "nba_imputer.pkl"))
    calibrator = joblib.load(os.path.join(out_dir, "nba_isotonic.pkl"))

    with open(os.path.join(out_dir, "nba_artifacts_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]

    priors = mp.load_franchise_priors(franchise)

    games = mp.build_game_rows(raw_logs)
    state = build_state_snapshot(games)

    abbr_to_id = {r["team_abbreviation"]: int(r["team_id"]) for _, r in teams_df.iterrows()}

    # Matchups from user screenshot
    matchups = [
        {"team_a": "BOS", "team_b": "OKC", "price_a_c": 31.0, "price_b_c": 70.0},
        {"team_a": "CHI", "team_b": "LAL", "price_a_c": 18.0, "price_b_c": 83.0},
    ]

    game_date = pd.Timestamp(datetime.now().date())
    season_label = infer_season_label(game_date)

    rows = []
    for m in matchups:
        a = m["team_a"]
        b = m["team_b"]
        if a not in abbr_to_id or b not in abbr_to_id:
            rows.append({"team_a": a, "team_b": b, "status": "MISSING_TEAM_ABBR"})
            continue

        pa_raw, pa_cal = symmetric_prob(
            abbr_to_id[a],
            abbr_to_id[b],
            game_date,
            season_label,
            priors,
            state,
            model,
            imputer,
            calibrator,
            feature_cols,
        )

        pb_raw = 1.0 - pa_raw
        pb_cal = 1.0 - pa_cal

        ma = m["price_a_c"] / 100.0
        mb = m["price_b_c"] / 100.0

        # Direct market comparison
        edge_a_c = (pa_cal - ma) * 100.0
        edge_b_c = (pb_cal - mb) * 100.0

        # No-vig comparison
        total = ma + mb
        ma_nv = ma / total
        mb_nv = mb / total
        edge_a_nv_c = (pa_cal - ma_nv) * 100.0
        edge_b_nv_c = (pb_cal - mb_nv) * 100.0

        if edge_a_c >= edge_b_c:
            best_side = a
            best_edge_c = edge_a_c
            best_price_c = m["price_a_c"]
            best_prob = pa_cal
        else:
            best_side = b
            best_edge_c = edge_b_c
            best_price_c = m["price_b_c"]
            best_prob = pb_cal

        action = "BUY" if best_edge_c >= EDGE_THRESHOLD_CENTS else "DONT_BUY"
        roi_pct = (best_edge_c / best_price_c) * 100.0 if best_price_c > 0 else np.nan

        rows.append(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "game_date": str(game_date.date()),
                "season_label": season_label,
                "team_a": a,
                "team_b": b,
                "price_a_cents": m["price_a_c"],
                "price_b_cents": m["price_b_c"],
                "overround_pct": (ma + mb) * 100.0,
                "prob_a_raw": pa_raw,
                "prob_b_raw": pb_raw,
                "prob_a_cal": pa_cal,
                "prob_b_cal": pb_cal,
                "fair_a_cents": pa_cal * 100.0,
                "fair_b_cents": pb_cal * 100.0,
                "edge_a_cents": edge_a_c,
                "edge_b_cents": edge_b_c,
                "edge_a_no_vig_cents": edge_a_nv_c,
                "edge_b_no_vig_cents": edge_b_nv_c,
                "best_side": best_side,
                "best_edge_cents": best_edge_c,
                "best_price_cents": best_price_c,
                "expected_roi_pct": roi_pct,
                "action": action,
                "status": "OK",
            }
        )

    report = pd.DataFrame(rows)

    xlsx_path = os.path.join(out_dir, "nba_ev_report_attached_games.xlsx")
    csv_path = os.path.join(out_dir, "nba_ev_report_attached_games.csv")

    report.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        report.to_excel(writer, sheet_name="ev_report", index=False)
        report[[
            "team_a", "team_b", "prob_a_cal", "prob_b_cal", "price_a_cents", "price_b_cents", "fair_a_cents", "fair_b_cents", "edge_a_cents", "edge_b_cents", "best_side", "action"
        ]].to_excel(writer, sheet_name="quick_view", index=False)

    print(f"Saved: {xlsx_path}")
    print(f"Saved: {csv_path}")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()

