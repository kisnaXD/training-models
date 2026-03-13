#!/usr/bin/env python3
"""Soccer EV CLI (3-way probabilities: A / Draw / B)."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from difflib import get_close_matches

import numpy as np
import pandas as pd
import joblib

import soccer_ev_cli as base_ev

EDGE_THRESHOLD_CENTS = 3.0


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)


def apply_temperature(prob, t):
    eps = 1e-12
    logits = np.log(np.clip(prob, eps, 1.0))
    return softmax(logits / max(float(t), 1e-6))


def infer_comp(title: str):
    return base_ev.infer_comp(title)


def resolve_team(name: str, team_pool):
    n = str(name).strip()
    if n in team_pool:
        return n
    lower_map = {x.lower(): x for x in team_pool}
    if n.lower() in lower_map:
        return lower_map[n.lower()]
    m = get_close_matches(n, team_pool, n=1, cutoff=0.62)
    return m[0] if m else None


def run(matchups_df: pd.DataFrame, base_dir: str, output_path: str | None = None):
    data_dir = os.path.join(base_dir, "soccer_data")
    out_dir = os.path.join(base_dir, "soccer_outputs_3way")
    if not os.path.exists(out_dir):
        raise FileNotFoundError(f"Missing {out_dir}. Run soccer_model_pipeline_3way.py first.")

    matches = pd.read_csv(os.path.join(data_dir, "matches.csv"))
    matches["match_date"] = pd.to_datetime(matches["match_date"], errors="coerce")
    matches = matches.dropna(subset=["match_date", "home_goals", "away_goals"])
    matches = matches.sort_values(["match_date", "competition_code", "home_team", "away_team"]).reset_index(drop=True)

    model = joblib.load(os.path.join(out_dir, "soccer_xgb_model_3way.pkl"))
    imputer = joblib.load(os.path.join(out_dir, "soccer_imputer_3way.pkl"))
    with open(os.path.join(out_dir, "soccer_artifacts_meta_3way.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    encoded_columns = meta["encoded_columns"]
    temperature = float(meta.get("temperature", 1.0))

    all_teams = sorted(set(matches["home_team"].astype(str)) | set(matches["away_team"].astype(str)))
    state_cache = {}

    rows = []
    for _, m in matchups_df.iterrows():
        a_in = str(m.get("team_a", "")).strip()
        b_in = str(m.get("team_b", "")).strip()
        game_date = pd.Timestamp(m["game_date"]) if "game_date" in m and pd.notna(m["game_date"]) else pd.Timestamp(datetime.now().date())
        comp = str(m.get("competition_code", "")).strip().upper()
        comp_name = str(m.get("competition_name", "")).strip()
        title = str(m.get("title", "")).strip()
        if not comp or not comp_name:
            c2, n2 = infer_comp(title)
            comp = comp or c2
            comp_name = comp_name or n2

        price_a = float(m["price_a_cents"]) if "price_a_cents" in m and pd.notna(m["price_a_cents"]) else np.nan
        price_b = float(m["price_b_cents"]) if "price_b_cents" in m and pd.notna(m["price_b_cents"]) else np.nan
        price_draw = float(m["price_draw_cents"]) if "price_draw_cents" in m and pd.notna(m["price_draw_cents"]) else np.nan

        a = resolve_team(a_in, all_teams)
        b = resolve_team(b_in, all_teams)
        if a is None or b is None:
            rows.append({"team_a_input": a_in, "team_b_input": b_in, "status": "MISSING_TEAM"})
            continue

        season_start = int(game_date.year if game_date.month >= 7 else game_date.year - 1)
        key = str(game_date.date())
        if key not in state_cache:
            state_cache[key] = base_ev.build_state_until(matches, game_date)
        state = state_cache[key]

        row_ab = base_ev.make_row(a, b, comp, comp_name, game_date, season_start, state, team_a_is_home=1)
        row_ba = base_ev.make_row(b, a, comp, comp_name, game_date, season_start, state, team_a_is_home=0)

        X = pd.DataFrame(
            [{c: row_ab.get(c, np.nan) for c in feature_cols}, {c: row_ba.get(c, np.nan) for c in feature_cols}],
            columns=feature_cols,
        )
        X_enc = pd.get_dummies(X, columns=["competition_code", "competition_name", "team_a", "team_b"], dummy_na=False, dtype=np.uint8)
        X_enc = X_enc.reindex(columns=encoded_columns, fill_value=0)
        Xi = imputer.transform(X_enc)

        p_ab = apply_temperature(model.predict_proba(Xi[[0]]), temperature)[0]  # [A, D, B]
        p_ba = apply_temperature(model.predict_proba(Xi[[1]]), temperature)[0]  # [B, D, A] from original orientation

        p_a = 0.5 * (p_ab[0] + p_ba[2])
        p_d = 0.5 * (p_ab[1] + p_ba[1])
        p_b = 0.5 * (p_ab[2] + p_ba[0])

        rec = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "game_date": str(game_date.date()),
            "competition_code": comp,
            "competition_name": comp_name,
            "team_a": a,
            "team_b": b,
            "price_a_cents": price_a,
            "price_draw_cents": price_draw,
            "price_b_cents": price_b,
            "prob_a_cal": p_a,
            "prob_draw_cal": p_d,
            "prob_b_cal": p_b,
            "fair_a_cents": p_a * 100.0,
            "fair_draw_cents": p_d * 100.0,
            "fair_b_cents": p_b * 100.0,
            "status": "OK",
        }

        # If only 2-way prices are available on market, evaluate conditional no-draw edge.
        if pd.notna(price_a) and pd.notna(price_b) and pd.isna(price_draw):
            denom = max(1e-12, 1.0 - p_d)
            p_a_2w = p_a / denom
            p_b_2w = p_b / denom
            ma = price_a / 100.0
            mb = price_b / 100.0
            ea = (p_a_2w - ma) * 100.0
            eb = (p_b_2w - mb) * 100.0
            if ea >= eb:
                best_side, best_edge, best_price = a, ea, price_a
            else:
                best_side, best_edge, best_price = b, eb, price_b
            action = "BUY" if best_edge >= EDGE_THRESHOLD_CENTS else "DONT_BUY"
            rec.update(
                {
                    "model_mode": "2way_conditional_nodraw",
                    "prob_a_cond_nodraw": p_a_2w,
                    "prob_b_cond_nodraw": p_b_2w,
                    "edge_a_cents": ea,
                    "edge_b_cents": eb,
                    "edge_draw_cents": np.nan,
                    "best_side": best_side,
                    "best_edge_cents": best_edge,
                    "best_price_cents": best_price,
                    "action": action,
                }
            )
        elif pd.notna(price_a) and pd.notna(price_b) and pd.notna(price_draw):
            ma = price_a / 100.0
            md = price_draw / 100.0
            mb = price_b / 100.0
            ea = (p_a - ma) * 100.0
            ed = (p_d - md) * 100.0
            eb = (p_b - mb) * 100.0
            best_side, best_edge, best_price = a, ea, price_a
            if ed > best_edge:
                best_side, best_edge, best_price = "DRAW", ed, price_draw
            if eb > best_edge:
                best_side, best_edge, best_price = b, eb, price_b
            action = "BUY" if best_edge >= EDGE_THRESHOLD_CENTS else "DONT_BUY"
            rec.update(
                {
                    "model_mode": "3way_direct",
                    "prob_a_cond_nodraw": np.nan,
                    "prob_b_cond_nodraw": np.nan,
                    "edge_a_cents": ea,
                    "edge_b_cents": eb,
                    "edge_draw_cents": ed,
                    "best_side": best_side,
                    "best_edge_cents": best_edge,
                    "best_price_cents": best_price,
                    "action": action,
                }
            )
        else:
            rec.update(
                {
                    "model_mode": "price_missing",
                    "prob_a_cond_nodraw": np.nan,
                    "prob_b_cond_nodraw": np.nan,
                    "edge_a_cents": np.nan,
                    "edge_b_cents": np.nan,
                    "edge_draw_cents": np.nan,
                    "best_side": "",
                    "best_edge_cents": np.nan,
                    "best_price_cents": np.nan,
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
    parser = argparse.ArgumentParser(description="Soccer EV CLI 3-way")
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--team-a", default=None)
    parser.add_argument("--team-b", default=None)
    parser.add_argument("--price-a-cents", type=float, default=None)
    parser.add_argument("--price-b-cents", type=float, default=None)
    parser.add_argument("--price-draw-cents", type=float, default=None)
    parser.add_argument("--game-date", default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    if args.input_csv:
        run(pd.read_csv(os.path.abspath(args.input_csv)), base_dir, output_path=args.output_csv)
        return

    if not all([args.team_a, args.team_b, args.price_a_cents is not None, args.price_b_cents is not None]):
        raise ValueError("Provide team_a/team_b and price_a/price_b (and optional draw) or use --input-csv.")

    row = {
        "team_a": args.team_a,
        "team_b": args.team_b,
        "price_a_cents": args.price_a_cents,
        "price_b_cents": args.price_b_cents,
    }
    if args.price_draw_cents is not None:
        row["price_draw_cents"] = args.price_draw_cents
    if args.game_date:
        row["game_date"] = args.game_date
    if args.title:
        row["title"] = args.title
    run(pd.DataFrame([row]), base_dir, output_path=args.output_csv)


if __name__ == "__main__":
    main()
