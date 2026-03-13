#!/usr/bin/env python3
"""NBA EV CLI (single matchup or batch CSV). Prices optional in batch mode."""

import argparse
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
    start = dt.year if dt.month >= 10 else dt.year - 1
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

        h_elo = elo[home]
        a_elo = elo[away]
        h_efg = (g["home_fgm"] + 0.5 * g["home_fg3m"]) / max(g["home_fga"], 1.0)
        a_efg = (g["away_fgm"] + 0.5 * g["away_fg3m"]) / max(g["away_fga"], 1.0)

        team_hist[home].append({"win": home_win, "pts": g["home_pts"], "opp_pts": g["away_pts"], "efg": h_efg, "tov": g["home_tov"], "reb": g["home_reb"], "ast": g["home_ast"], "opp_elo": a_elo})
        team_hist[away].append({"win": 1 - home_win, "pts": g["away_pts"], "opp_pts": g["home_pts"], "efg": a_efg, "tov": g["away_tov"], "reb": g["away_reb"], "ast": g["away_ast"], "opp_elo": h_elo})

        season_record[(home, season)]["w"] += home_win
        season_record[(home, season)]["l"] += (1 - home_win)
        season_record[(away, season)]["w"] += (1 - home_win)
        season_record[(away, season)]["l"] += home_win

        last_game_date[home] = g["game_date"]
        last_game_date[away] = g["game_date"]

        nh, na = mp.elo_update(h_elo, a_elo, bool(home_win), mov, playoffs=playoffs)
        elo[home] = nh
        elo[away] = na

    return {"elo": elo, "team_hist": team_hist, "last_game_date": last_game_date, "season_record": season_record}


def make_feature_row(home_id, away_id, game_date, season_label, priors, state):
    season_start = mp.parse_season_start(season_label)
    elo = state["elo"]
    team_hist = state["team_hist"]
    last_game_date = state["last_game_date"]
    season_record = state["season_record"]

    h_elo = elo[home_id]
    a_elo = elo[away_id]
    h_rest = (game_date - last_game_date[home_id]).days if home_id in last_game_date else 5
    a_rest = (game_date - last_game_date[away_id]).days if away_id in last_game_date else 5

    h5 = mp.summarize_hist(team_hist[home_id], 5)
    h10 = mp.summarize_hist(team_hist[home_id], 10)
    h20 = mp.summarize_hist(team_hist[home_id], 20)
    a5 = mp.summarize_hist(team_hist[away_id], 5)
    a10 = mp.summarize_hist(team_hist[away_id], 10)
    a20 = mp.summarize_hist(team_hist[away_id], 20)

    hr = season_record[(home_id, season_label)]
    ar = season_record[(away_id, season_label)]
    hg = hr["w"] + hr["l"]
    ag = ar["w"] + ar["l"]
    hwr = (hr["w"] / hg) if hg > 0 else 0.5
    awr = (ar["w"] / ag) if ag > 0 else 0.5

    hp = priors.get((home_id, season_start - 1), {"prior_win_pct": 0.5, "prior_playoff_wins": 0.0})
    ap = priors.get((away_id, season_start - 1), {"prior_win_pct": 0.5, "prior_playoff_wins": 0.0})

    return {
        "season_start": season_start,
        "home_team_id": home_id,
        "away_team_id": away_id,
        "playoffs": 0,
        "elo_diff": h_elo - a_elo,
        "home_elo": h_elo,
        "away_elo": a_elo,
        "rest_diff": h_rest - a_rest,
        "home_rest": h_rest,
        "away_rest": a_rest,
        "home_b2b": int(h_rest <= 1),
        "away_b2b": int(a_rest <= 1),
        "season_wr_diff": hwr - awr,
        "home_season_wr": hwr,
        "away_season_wr": awr,
        "home_season_games": hg,
        "away_season_games": ag,
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


def symmetric_probs(team_a_id, team_b_id, game_date, season_label, priors, state, model, imputer, calibrator, feature_cols):
    row_ab = make_feature_row(team_a_id, team_b_id, game_date, season_label, priors, state)
    row_ba = make_feature_row(team_b_id, team_a_id, game_date, season_label, priors, state)

    Xab = pd.DataFrame([{c: row_ab.get(c, np.nan) for c in feature_cols}])[feature_cols]
    Xba = pd.DataFrame([{c: row_ba.get(c, np.nan) for c in feature_cols}])[feature_cols]

    p_ab_raw = float(model.predict_proba(imputer.transform(Xab))[0, 1])
    p_ba_raw = float(model.predict_proba(imputer.transform(Xba))[0, 1])
    p_ab_cal = float(calibrator.transform([p_ab_raw])[0])
    p_ba_cal = float(calibrator.transform([p_ba_raw])[0])

    return 0.5 * (p_ab_raw + (1.0 - p_ba_raw)), 0.5 * (p_ab_cal + (1.0 - p_ba_cal))


def run(matchups_df, base_dir, output_path=None):
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

    games = mp.build_game_rows(raw_logs)
    state = build_state_snapshot(games)
    priors = mp.load_franchise_priors(franchise)
    abbr_to_id = {r["team_abbreviation"]: int(r["team_id"]) for _, r in teams_df.iterrows()}

    rows = []
    for _, m in matchups_df.iterrows():
        a = str(m["team_a"]).upper().strip()
        b = str(m["team_b"]).upper().strip()
        gdate = pd.Timestamp(m["game_date"]) if "game_date" in m and pd.notna(m["game_date"]) else pd.Timestamp(datetime.now().date())
        season_label = infer_season_label(gdate)

        pa_c = float(m["price_a_cents"]) if "price_a_cents" in m and pd.notna(m["price_a_cents"]) else np.nan
        pb_c = float(m["price_b_cents"]) if "price_b_cents" in m and pd.notna(m["price_b_cents"]) else np.nan

        if a not in abbr_to_id or b not in abbr_to_id:
            rows.append({"team_a": a, "team_b": b, "status": "MISSING_TEAM_ABBR"})
            continue

        p_raw, p_cal = symmetric_probs(abbr_to_id[a], abbr_to_id[b], gdate, season_label, priors, state, model, imputer, calibrator, feature_cols)
        p_b_raw = 1.0 - p_raw
        p_b_cal = 1.0 - p_cal

        rec = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "game_date": str(gdate.date()),
            "season_label": season_label,
            "team_a": a,
            "team_b": b,
            "price_a_cents": pa_c,
            "price_b_cents": pb_c,
            "prob_a_raw": p_raw,
            "prob_b_raw": p_b_raw,
            "prob_a_cal": p_cal,
            "prob_b_cal": p_b_cal,
            "fair_a_cents": p_cal * 100.0,
            "fair_b_cents": p_b_cal * 100.0,
            "status": "OK",
        }

        if pd.notna(pa_c) and pd.notna(pb_c):
            ma = pa_c / 100.0
            mb = pb_c / 100.0
            e_a = (p_cal - ma) * 100.0
            e_b = (p_b_cal - mb) * 100.0
            total = ma + mb
            ma_nv = ma / total
            mb_nv = mb / total
            e_a_nv = (p_cal - ma_nv) * 100.0
            e_b_nv = (p_b_cal - mb_nv) * 100.0
            if e_a >= e_b:
                best_side, best_edge, best_price = a, e_a, pa_c
            else:
                best_side, best_edge, best_price = b, e_b, pb_c
            action = "BUY" if best_edge >= EDGE_THRESHOLD_CENTS else "DONT_BUY"
            roi = (best_edge / best_price) * 100.0 if best_price > 0 else np.nan

            rec.update({
                "overround_pct": (ma + mb) * 100.0,
                "edge_a_cents": e_a,
                "edge_b_cents": e_b,
                "edge_a_no_vig_cents": e_a_nv,
                "edge_b_no_vig_cents": e_b_nv,
                "best_side": best_side,
                "best_edge_cents": best_edge,
                "best_price_cents": best_price,
                "expected_roi_pct": roi,
                "action": action,
            })
        else:
            rec.update({
                "overround_pct": np.nan,
                "edge_a_cents": np.nan,
                "edge_b_cents": np.nan,
                "edge_a_no_vig_cents": np.nan,
                "edge_b_no_vig_cents": np.nan,
                "best_side": "",
                "best_edge_cents": np.nan,
                "best_price_cents": np.nan,
                "expected_roi_pct": np.nan,
                "action": "PRICE_MISSING",
            })

        rows.append(rec)

    out = pd.DataFrame(rows)

    if output_path:
        output_path = os.path.abspath(output_path)
        out.to_csv(output_path, index=False)
        xlsx_path = os.path.splitext(output_path)[0] + ".xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            out.to_excel(writer, sheet_name="report", index=False)
            quick_cols = ["team_a", "team_b", "prob_a_cal", "prob_b_cal", "price_a_cents", "price_b_cents", "fair_a_cents", "fair_b_cents", "edge_a_cents", "edge_b_cents", "best_side", "action"]
            out[[c for c in quick_cols if c in out.columns]].to_excel(writer, sheet_name="quick_view", index=False)
        print(f"Saved CSV: {output_path}")
        print(f"Saved XLSX: {xlsx_path}")

    print(out.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="NBA EV CLI")
    parser.add_argument("--team-a", help="Team A abbreviation")
    parser.add_argument("--team-b", help="Team B abbreviation")
    parser.add_argument("--price-a-cents", type=float, help="Team A market price cents")
    parser.add_argument("--price-b-cents", type=float, help="Team B market price cents")
    parser.add_argument("--game-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--input-csv", default=None, help="Batch CSV: team_a,team_b[,price_a_cents,price_b_cents,game_date]")
    parser.add_argument("--output-csv", default=None, help="Output CSV path")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    if args.input_csv:
        df = pd.read_csv(os.path.abspath(args.input_csv))
        required = {"team_a", "team_b"}
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in input CSV: {missing}")
        run(df, base_dir=base_dir, output_path=args.output_csv)
        return

    if not all([args.team_a, args.team_b]):
        raise ValueError("Single mode requires --team-a and --team-b")

    row = {"team_a": args.team_a, "team_b": args.team_b}
    if args.price_a_cents is not None:
        row["price_a_cents"] = args.price_a_cents
    if args.price_b_cents is not None:
        row["price_b_cents"] = args.price_b_cents
    if args.game_date:
        row["game_date"] = args.game_date

    run(pd.DataFrame([row]), base_dir=base_dir, output_path=args.output_csv)


if __name__ == "__main__":
    main()
