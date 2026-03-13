#!/usr/bin/env python3
"""Soccer EV CLI using cleaned mirrored + calibrated artifacts."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict, deque
from datetime import datetime
from difflib import get_close_matches
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

ELO_BASE = 1500.0
HOME_ADV = 60.0
EDGE_THRESHOLD_CENTS = 3.0


def implied_prob_decimal(odds):
    if pd.isna(odds):
        return np.nan
    o = float(odds)
    if o <= 1.0:
        return np.nan
    return 1.0 / o


def no_vig(p1, p2, p3):
    vals = [p1, p2, p3]
    if any(pd.isna(v) for v in vals):
        return (np.nan, np.nan, np.nan)
    s = p1 + p2 + p3
    if s <= 0:
        return (np.nan, np.nan, np.nan)
    return (p1 / s, p2 / s, p3 / s)


def summarize(hist, n):
    h = list(hist)[-n:]
    if len(h) == 0:
        return {"wr": 0.5, "gf": 1.4, "ga": 1.4, "margin": 0.0, "opp_elo": ELO_BASE, "count": 0}
    return {
        "wr": float(np.mean([x["win"] for x in h])),
        "gf": float(np.mean([x["gf"] for x in h])),
        "ga": float(np.mean([x["ga"] for x in h])),
        "margin": float(np.mean([x["margin"] for x in h])),
        "opp_elo": float(np.mean([x["opp_elo"] for x in h])),
        "count": len(h),
    }


def elo_expect(home_elo, away_elo):
    return 1.0 / (1.0 + 10.0 ** ((away_elo - (home_elo + HOME_ADV)) / 400.0))


def elo_update(home_elo, away_elo, home_win, mov, is_cup=False):
    exp_home = elo_expect(home_elo, away_elo)
    k = 24.0 if is_cup else 20.0
    mov_mult = ((abs(mov) + 1.0) ** 0.8) / (7.5 + 0.006 * abs(home_elo - away_elo))
    delta = k * mov_mult * ((1.0 if home_win else 0.0) - exp_home)
    return home_elo + delta, away_elo - delta


def infer_comp(title: str) -> Tuple[str, str]:
    t = str(title or "").lower()
    if "champions league" in t:
        return "CL", "UEFA Champions League"
    if "europa league" in t:
        return "EL", "UEFA Europa League"
    if "conference league" in t:
        return "EC", "UEFA Conference League"
    return "E0", "Premier League"


def resolve_team(name: str, team_pool):
    n = str(name).strip()
    if n in team_pool:
        return n
    lower_map = {x.lower(): x for x in team_pool}
    if n.lower() in lower_map:
        return lower_map[n.lower()]
    m = get_close_matches(n, team_pool, n=1, cutoff=0.62)
    return m[0] if m else None


def build_state_until(matches: pd.DataFrame, cutoff_date: pd.Timestamp):
    overall_elo = defaultdict(lambda: ELO_BASE)
    comp_elo = defaultdict(lambda: ELO_BASE)
    overall_hist = defaultdict(lambda: deque(maxlen=60))
    comp_hist = defaultdict(lambda: deque(maxlen=40))
    last_date = {}
    comp_season_record = defaultdict(lambda: {"w": 0, "l": 0})

    for _, g in matches.iterrows():
        gd = pd.Timestamp(g["match_date"])
        if gd >= cutoff_date:
            break

        home = g["home_team"]
        away = g["away_team"]
        comp = g["competition_code"]
        season = int(g["season_start"])
        hg = float(g["home_goals"])
        ag = float(g["away_goals"])
        home_win = int(hg > ag)
        draw = int(hg == ag)
        away_win = int(hg < ag)
        mov = hg - ag

        h_elo_o = overall_elo[home]
        a_elo_o = overall_elo[away]
        h_elo_c = comp_elo[(home, comp)]
        a_elo_c = comp_elo[(away, comp)]

        overall_hist[home].append({"win": home_win, "gf": hg, "ga": ag, "margin": mov, "opp_elo": a_elo_o})
        overall_hist[away].append({"win": away_win, "gf": ag, "ga": hg, "margin": -mov, "opp_elo": h_elo_o})
        comp_hist[(home, comp)].append({"win": home_win, "gf": hg, "ga": ag, "margin": mov, "opp_elo": a_elo_c})
        comp_hist[(away, comp)].append({"win": away_win, "gf": ag, "ga": hg, "margin": -mov, "opp_elo": h_elo_c})

        comp_season_record[(home, comp, season)]["w"] += home_win
        comp_season_record[(home, comp, season)]["l"] += away_win
        comp_season_record[(away, comp, season)]["w"] += away_win
        comp_season_record[(away, comp, season)]["l"] += home_win
        last_date[home] = gd
        last_date[away] = gd

        is_cup = comp in ["CL", "EL", "EC"]
        no_draw_home_win = bool(home_win) if draw == 0 else False
        new_ho, new_ao = elo_update(h_elo_o, a_elo_o, no_draw_home_win, mov, is_cup=is_cup)
        new_hc, new_ac = elo_update(h_elo_c, a_elo_c, no_draw_home_win, mov, is_cup=is_cup)
        overall_elo[home], overall_elo[away] = new_ho, new_ao
        comp_elo[(home, comp)], comp_elo[(away, comp)] = new_hc, new_ac

    return {
        "overall_elo": overall_elo,
        "comp_elo": comp_elo,
        "overall_hist": overall_hist,
        "comp_hist": comp_hist,
        "last_date": last_date,
        "comp_season_record": comp_season_record,
    }


def make_row(team_a, team_b, comp, comp_name, game_date, season_start, state, team_a_is_home=1):
    overall_elo = state["overall_elo"]
    comp_elo = state["comp_elo"]
    overall_hist = state["overall_hist"]
    comp_hist = state["comp_hist"]
    last_date = state["last_date"]
    comp_season_record = state["comp_season_record"]

    a_elo_o = overall_elo[team_a]
    b_elo_o = overall_elo[team_b]
    a_elo_c = comp_elo[(team_a, comp)]
    b_elo_c = comp_elo[(team_b, comp)]
    a_rest = (game_date - last_date[team_a]).days if team_a in last_date else 7
    b_rest = (game_date - last_date[team_b]).days if team_b in last_date else 7

    a5o = summarize(overall_hist[team_a], 5)
    a10o = summarize(overall_hist[team_a], 10)
    b5o = summarize(overall_hist[team_b], 5)
    b10o = summarize(overall_hist[team_b], 10)
    a5c = summarize(comp_hist[(team_a, comp)], 5)
    a10c = summarize(comp_hist[(team_a, comp)], 10)
    b5c = summarize(comp_hist[(team_b, comp)], 5)
    b10c = summarize(comp_hist[(team_b, comp)], 10)

    ar = comp_season_record[(team_a, comp, season_start)]
    br = comp_season_record[(team_b, comp, season_start)]
    ag_n = ar["w"] + ar["l"]
    bg_n = br["w"] + br["l"]
    a_wr = ar["w"] / ag_n if ag_n > 0 else 0.5
    b_wr = br["w"] / bg_n if bg_n > 0 else 0.5

    return {
        "season_start": int(season_start),
        "competition_code": comp,
        "competition_name": comp_name,
        "is_uefa_comp": int(comp in ["CL", "EL", "EC"]),
        "month": int(game_date.month),
        "weekday": int(game_date.dayofweek),
        "team_a": team_a,
        "team_b": team_b,
        "team_a_is_home": int(team_a_is_home),
        "elo_overall_diff": a_elo_o - b_elo_o,
        "elo_comp_diff": a_elo_c - b_elo_c,
        "rest_diff": a_rest - b_rest,
        "team_a_rest": a_rest,
        "team_b_rest": b_rest,
        "team_a_b2b": int(a_rest <= 3),
        "team_b_b2b": int(b_rest <= 3),
        "season_wr_comp_diff": a_wr - b_wr,
        "wr5_overall_diff": a5o["wr"] - b5o["wr"],
        "wr10_overall_diff": a10o["wr"] - b10o["wr"],
        "margin10_overall_diff": a10o["margin"] - b10o["margin"],
        "gf10_overall_diff": a10o["gf"] - b10o["gf"],
        "ga10_overall_diff": a10o["ga"] - b10o["ga"],
        "wr5_comp_diff": a5c["wr"] - b5c["wr"],
        "wr10_comp_diff": a10c["wr"] - b10c["wr"],
        "margin10_comp_diff": a10c["margin"] - b10c["margin"],
        "gf10_comp_diff": a10c["gf"] - b10c["gf"],
        "ga10_comp_diff": a10c["ga"] - b10c["ga"],
        "opp_elo10_overall_diff": a10o["opp_elo"] - b10o["opp_elo"],
        "hist10_overall_a": a10o["count"],
        "hist10_overall_b": b10o["count"],
        "hist10_comp_a": a10c["count"],
        "hist10_comp_b": b10c["count"],
        "b365_prob_a": np.nan,
        "b365_prob_draw": np.nan,
        "b365_prob_b": np.nan,
        "b365_prob_a_novig": np.nan,
        "b365_prob_draw_novig": np.nan,
        "b365_prob_b_novig": np.nan,
    }


def run(matchups_df: pd.DataFrame, base_dir: str, output_path: str | None = None):
    data_dir = os.path.join(base_dir, "soccer_data")
    out_dir_clean = os.path.join(base_dir, "soccer_outputs_clean")
    out_dir_fallback = os.path.join(base_dir, "soccer_outputs")
    out_dir = out_dir_clean if os.path.exists(out_dir_clean) else out_dir_fallback

    matches = pd.read_csv(os.path.join(data_dir, "matches.csv"))
    matches["match_date"] = pd.to_datetime(matches["match_date"], errors="coerce")
    matches = matches.dropna(subset=["match_date", "home_goals", "away_goals"])
    matches = matches.sort_values(["match_date", "competition_code", "home_team", "away_team"]).reset_index(drop=True)

    model = joblib.load(os.path.join(out_dir, "soccer_xgb_model.pkl"))
    imputer = joblib.load(os.path.join(out_dir, "soccer_imputer.pkl"))
    iso = joblib.load(os.path.join(out_dir, "soccer_isotonic.pkl"))
    with open(os.path.join(out_dir, "soccer_artifacts_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    encoded_columns = meta["encoded_columns"]

    all_teams = sorted(set(matches["home_team"].astype(str).tolist()) | set(matches["away_team"].astype(str).tolist()))
    state_cache: Dict[str, dict] = {}

    rows = []
    for _, m in matchups_df.iterrows():
        a_in = str(m["team_a"]).strip()
        b_in = str(m["team_b"]).strip()
        game_date = pd.Timestamp(m["game_date"]) if "game_date" in m and pd.notna(m["game_date"]) else pd.Timestamp(datetime.now().date())
        comp = str(m["competition_code"]).strip().upper() if "competition_code" in m and pd.notna(m["competition_code"]) else ""
        comp_name = str(m["competition_name"]).strip() if "competition_name" in m and pd.notna(m["competition_name"]) else ""
        title = str(m["title"]) if "title" in m and pd.notna(m["title"]) else ""
        if not comp or not comp_name:
            c2, n2 = infer_comp(title)
            if not comp:
                comp = c2
            if not comp_name:
                comp_name = n2

        price_a = float(m["price_a_cents"]) if "price_a_cents" in m and pd.notna(m["price_a_cents"]) else np.nan
        price_b = float(m["price_b_cents"]) if "price_b_cents" in m and pd.notna(m["price_b_cents"]) else np.nan

        a = resolve_team(a_in, all_teams)
        b = resolve_team(b_in, all_teams)
        if a is None or b is None:
            rows.append({"team_a_input": a_in, "team_b_input": b_in, "status": "MISSING_TEAM"})
            continue

        season_start = int(game_date.year if game_date.month >= 7 else game_date.year - 1)
        k = str(game_date.date())
        if k not in state_cache:
            state_cache[k] = build_state_until(matches, game_date)
        state = state_cache[k]

        row_ab = make_row(a, b, comp, comp_name, game_date, season_start, state, team_a_is_home=1)
        row_ba = make_row(b, a, comp, comp_name, game_date, season_start, state, team_a_is_home=0)

        X = pd.DataFrame([{c: row_ab.get(c, np.nan) for c in feature_cols}, {c: row_ba.get(c, np.nan) for c in feature_cols}], columns=feature_cols)
        X_enc = pd.get_dummies(X, columns=["competition_code", "competition_name", "team_a", "team_b"], dummy_na=False, dtype=np.uint8)
        for c in encoded_columns:
            if c not in X_enc.columns:
                X_enc[c] = 0
        X_enc = X_enc[encoded_columns]
        Xi = imputer.transform(X_enc)

        p_ab_raw = float(model.predict_proba(Xi[[0]])[0, 1])
        p_ba_raw = float(model.predict_proba(Xi[[1]])[0, 1])
        p_ab_cal = float(iso.transform([p_ab_raw])[0])
        p_ba_cal = float(iso.transform([p_ba_raw])[0])

        p_a_raw = 0.5 * (p_ab_raw + (1.0 - p_ba_raw))
        p_a_cal = 0.5 * (p_ab_cal + (1.0 - p_ba_cal))
        p_b_raw = 1.0 - p_a_raw
        p_b_cal = 1.0 - p_a_cal

        rec = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "game_date": str(game_date.date()),
            "competition_code": comp,
            "competition_name": comp_name,
            "team_a": a,
            "team_b": b,
            "price_a_cents": price_a,
            "price_b_cents": price_b,
            "prob_a_raw": p_a_raw,
            "prob_b_raw": p_b_raw,
            "prob_a_cal": p_a_cal,
            "prob_b_cal": p_b_cal,
            "fair_a_cents": p_a_cal * 100.0,
            "fair_b_cents": p_b_cal * 100.0,
            "status": "OK",
        }

        if pd.notna(price_a) and pd.notna(price_b):
            ma = price_a / 100.0
            mb = price_b / 100.0
            ea = (p_a_cal - ma) * 100.0
            eb = (p_b_cal - mb) * 100.0
            if ea >= eb:
                best_side, best_edge, best_price = a, ea, price_a
            else:
                best_side, best_edge, best_price = b, eb, price_b
            action = "BUY" if best_edge >= EDGE_THRESHOLD_CENTS else "DONT_BUY"
            roi = (best_edge / best_price) * 100.0 if best_price > 0 else np.nan
            rec.update(
                {
                    "overround_pct": (ma + mb) * 100.0,
                    "edge_a_cents": ea,
                    "edge_b_cents": eb,
                    "best_side": best_side,
                    "best_edge_cents": best_edge,
                    "best_price_cents": best_price,
                    "expected_roi_pct": roi,
                    "action": action,
                }
            )
        else:
            rec.update(
                {
                    "overround_pct": np.nan,
                    "edge_a_cents": np.nan,
                    "edge_b_cents": np.nan,
                    "best_side": "",
                    "best_edge_cents": np.nan,
                    "best_price_cents": np.nan,
                    "expected_roi_pct": np.nan,
                    "action": "PRICE_MISSING",
                }
            )
        rows.append(rec)

    out = pd.DataFrame(rows)
    if output_path:
        out.to_csv(os.path.abspath(output_path), index=False)
    print(out.to_string(index=False))
    return out


def main():
    parser = argparse.ArgumentParser(description="Soccer EV CLI")
    parser.add_argument("--team-a")
    parser.add_argument("--team-b")
    parser.add_argument("--price-a-cents", type=float)
    parser.add_argument("--price-b-cents", type=float)
    parser.add_argument("--game-date", default=None)
    parser.add_argument("--competition-code", default=None)
    parser.add_argument("--competition-name", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    if args.input_csv:
        df = pd.read_csv(os.path.abspath(args.input_csv))
        run(df, base_dir, output_path=args.output_csv)
        return

    if not all([args.team_a, args.team_b, args.price_a_cents is not None, args.price_b_cents is not None]):
        raise ValueError("Provide --team-a --team-b --price-a-cents --price-b-cents or use --input-csv")

    row = {
        "team_a": args.team_a,
        "team_b": args.team_b,
        "price_a_cents": args.price_a_cents,
        "price_b_cents": args.price_b_cents,
    }
    if args.game_date:
        row["game_date"] = args.game_date
    if args.competition_code:
        row["competition_code"] = args.competition_code
    if args.competition_name:
        row["competition_name"] = args.competition_name
    if args.title:
        row["title"] = args.title
    run(pd.DataFrame([row]), base_dir, output_path=args.output_csv)


if __name__ == "__main__":
    main()

