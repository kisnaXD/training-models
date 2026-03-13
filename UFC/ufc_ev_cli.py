#!/usr/bin/env python3
"""UFC EV CLI (single matchup or batch CSV)."""

import argparse
import ast
import os
from datetime import datetime
from difflib import get_close_matches

import joblib
import numpy as np
import pandas as pd

import predict_v2 as p2

EDGE_THRESHOLD_CENTS = 3.0


def load_artifacts(base_dir):
    model = joblib.load(os.path.join(base_dir, "xgb_model_v2.pkl"))
    imputer = joblib.load(os.path.join(base_dir, "imputer_v2.pkl"))
    calibrator = joblib.load(os.path.join(base_dir, "isotonic_calibrator_v2.pkl"))
    arts = joblib.load(os.path.join(base_dir, "model_artifacts_v2.pkl"))

    elo_wc = {}
    for raw_key, val in arts["elo_wc"].items():
        try:
            elo_wc[ast.literal_eval(raw_key)] = val
        except Exception:
            continue

    all_fighters = sorted({n for n in (list(arts["elo_global"].keys()) + list(arts["fighter_phys"].keys())) if isinstance(n, str) and n.strip()})
    return model, imputer, calibrator, arts, elo_wc, all_fighters


def resolve_name(name, all_fighters):
    if name in all_fighters:
        return name
    m = get_close_matches(name, all_fighters, n=1, cutoff=0.6)
    return m[0] if m else None


def get_probs(f1, f2, wc, fight_date, model, imputer, calibrator, arts, elo_wc):
    p_ab_raw, p_ab_cal, _ = p2.direction_probability(f1, f2, wc, fight_date, model, imputer, calibrator, arts, elo_wc)
    p_ba_raw, p_ba_cal, _ = p2.direction_probability(f2, f1, wc, fight_date, model, imputer, calibrator, arts, elo_wc)

    p1_raw = 0.5 * (p_ab_raw + (1.0 - p_ba_raw))
    p1_cal = 0.5 * (p_ab_cal + (1.0 - p_ba_cal))
    return p1_raw, p1_cal


def run(df, base_dir, output_path=None):
    model, imputer, calibrator, arts, elo_wc, all_fighters = load_artifacts(base_dir)

    rows = []
    for _, r in df.iterrows():
        f1_in = str(r["fighter_a"]).strip()
        f2_in = str(r["fighter_b"]).strip()
        wc = str(r["weight_class"]) if "weight_class" in r and pd.notna(r["weight_class"]) and str(r["weight_class"]).strip() else "Lightweight"
        p1_c = float(r["price_a_cents"])
        p2_c = float(r["price_b_cents"])
        fdate = pd.Timestamp(r["fight_date"]) if "fight_date" in r and pd.notna(r["fight_date"]) else pd.Timestamp(datetime.now().date())

        f1 = resolve_name(f1_in, all_fighters)
        f2 = resolve_name(f2_in, all_fighters)
        if f1 is None or f2 is None:
            rows.append({"fighter_a_input": f1_in, "fighter_b_input": f2_in, "status": "MISSING_FIGHTER"})
            continue

        p1_raw, p1_cal = get_probs(f1, f2, wc, fdate, model, imputer, calibrator, arts, elo_wc)
        p2_raw = 1.0 - p1_raw
        p2_cal = 1.0 - p1_cal

        m1 = p1_c / 100.0
        m2 = p2_c / 100.0

        e1 = (p1_cal - m1) * 100.0
        e2 = (p2_cal - m2) * 100.0

        total = m1 + m2
        m1_nv = m1 / total
        m2_nv = m2 / total
        e1_nv = (p1_cal - m1_nv) * 100.0
        e2_nv = (p2_cal - m2_nv) * 100.0

        if e1 >= e2:
            best_side, best_edge, best_price = f1, e1, p1_c
        else:
            best_side, best_edge, best_price = f2, e2, p2_c

        action = "BUY" if best_edge >= EDGE_THRESHOLD_CENTS else "DONT_BUY"
        roi = (best_edge / best_price) * 100.0 if best_price > 0 else np.nan

        rows.append({
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "fight_date": str(fdate.date()),
            "fighter_a_input": f1_in,
            "fighter_b_input": f2_in,
            "fighter_a_resolved": f1,
            "fighter_b_resolved": f2,
            "weight_class": wc,
            "price_a_cents": p1_c,
            "price_b_cents": p2_c,
            "overround_pct": (m1 + m2) * 100.0,
            "prob_a_raw": p1_raw,
            "prob_b_raw": p2_raw,
            "prob_a_cal": p1_cal,
            "prob_b_cal": p2_cal,
            "fair_a_cents": p1_cal * 100.0,
            "fair_b_cents": p2_cal * 100.0,
            "edge_a_cents": e1,
            "edge_b_cents": e2,
            "edge_a_no_vig_cents": e1_nv,
            "edge_b_no_vig_cents": e2_nv,
            "best_side": best_side,
            "best_edge_cents": best_edge,
            "best_price_cents": best_price,
            "expected_roi_pct": roi,
            "action": action,
            "status": "OK",
        })

    out = pd.DataFrame(rows)

    if output_path:
        output_path = os.path.abspath(output_path)
        out.to_csv(output_path, index=False)
        xlsx_path = os.path.splitext(output_path)[0] + ".xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            out.to_excel(writer, sheet_name="ev_report", index=False)
            keep = ["fighter_a_resolved", "fighter_b_resolved", "weight_class", "prob_a_cal", "prob_b_cal", "price_a_cents", "price_b_cents", "fair_a_cents", "fair_b_cents", "edge_a_cents", "edge_b_cents", "best_side", "action"]
            cols = [c for c in keep if c in out.columns]
            out[cols].to_excel(writer, sheet_name="quick_view", index=False)
        print(f"Saved CSV: {output_path}")
        print(f"Saved XLSX: {xlsx_path}")

    print(out.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="UFC EV CLI")
    parser.add_argument("--fighter-a", help="Fighter A")
    parser.add_argument("--fighter-b", help="Fighter B")
    parser.add_argument("--weight-class", default="Lightweight", help="Weight class")
    parser.add_argument("--price-a-cents", type=float, help="Fighter A market price in cents")
    parser.add_argument("--price-b-cents", type=float, help="Fighter B market price in cents")
    parser.add_argument("--fight-date", default=None, help="Date YYYY-MM-DD")
    parser.add_argument("--input-csv", default=None, help="Batch input CSV: fighter_a,fighter_b,price_a_cents,price_b_cents[,weight_class,fight_date]")
    parser.add_argument("--output-csv", default=None, help="Where to save output CSV")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    if args.input_csv:
        df = pd.read_csv(os.path.abspath(args.input_csv))
        required = {"fighter_a", "fighter_b", "price_a_cents", "price_b_cents"}
        miss = [c for c in required if c not in df.columns]
        if miss:
            raise ValueError(f"Missing required columns in input CSV: {miss}")
        run(df, base_dir, output_path=args.output_csv)
        return

    if not all([args.fighter_a, args.fighter_b, args.price_a_cents is not None, args.price_b_cents is not None]):
        raise ValueError("For single-match mode provide --fighter-a --fighter-b --price-a-cents --price-b-cents")

    row = {
        "fighter_a": args.fighter_a,
        "fighter_b": args.fighter_b,
        "weight_class": args.weight_class,
        "price_a_cents": args.price_a_cents,
        "price_b_cents": args.price_b_cents,
    }
    if args.fight_date:
        row["fight_date"] = args.fight_date

    run(pd.DataFrame([row]), base_dir, output_path=args.output_csv)


if __name__ == "__main__":
    main()
