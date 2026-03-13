#!/usr/bin/env python3
"""
UFC Fight Predictor (v2)

New in v2:
- Uses isotonic-calibrated probabilities
- Symmetry averaging: P(A beats B) blended with 1 - P(B beats A)
- Fair price output (probability as tradable value)
- Robust odds parsing for decimal / implied % / American
"""

import sys
import ast
import argparse
from datetime import datetime
from difflib import get_close_matches

import joblib
import numpy as np
import pandas as pd


SEP = "=" * 70


def _safe(val):
    try:
        v = float(val)
        return 0.0 if np.isnan(v) else v
    except Exception:
        return 0.0


def _diff(a, b):
    if np.isnan(a) or np.isnan(b):
        return np.nan
    return a - b


def _age_at(dob, date):
    if pd.isna(dob) or date is None:
        return np.nan
    return (date - dob).days / 365.25


def _rolling(hist, n=None, elo_base=1500):
    h = hist[-n:] if n else hist
    total = len(h)
    if total == 0:
        return {
            "win_rate": 0.5,
            "ko_rate": 0.0,
            "sub_rate": 0.0,
            "dec_rate": 0.0,
            "avg_sig": 0.0,
            "avg_td": 0.0,
            "avg_ctrl": 0.0,
            "count": 0,
            "avg_opp_elo": float(elo_base),
        }

    wins = sum(x["w"] for x in h)
    return {
        "win_rate": wins / total,
        "ko_rate": sum(x["w"] and x["ko"] for x in h) / total,
        "sub_rate": sum(x["w"] and x["sub"] for x in h) / total,
        "dec_rate": sum(x["w"] and x["dec"] for x in h) / total,
        "avg_sig": float(np.mean([x["sig"] for x in h])),
        "avg_td": float(np.mean([x["td"] for x in h])),
        "avg_ctrl": float(np.mean([x["ctrl"] for x in h])),
        "count": total,
        "avg_opp_elo": float(np.mean([x.get("opp_elo", elo_base) for x in h])),
    }


def _streak(hist, won_val):
    streak = 0
    for x in reversed(hist):
        if bool(x["w"]) == won_val:
            streak += 1
        else:
            break
    return streak


def _h2h(fight_hist, me, opp):
    h = fight_hist.get(me, [])
    wins = sum(1 for x in h if x["opp"] == opp and x["w"])
    losses = sum(1 for x in h if x["opp"] == opp and not x["w"])
    return wins, losses


def american_to_implied(odds_text):
    o = float(odds_text.replace(" ", ""))
    if o < 0:
        return abs(o) / (abs(o) + 100.0)
    return 100.0 / (o + 100.0)


def parse_market_probability(raw):
    text = raw.strip().replace(",", "")
    if not text:
        raise ValueError("empty input")

    if text.endswith("%"):
        val = float(text[:-1])
        if not (0.0 < val < 100.0):
            raise ValueError("percent must be between 0 and 100")
        return val / 100.0

    if text.startswith("+") or text.startswith("-"):
        return american_to_implied(text)

    val = float(text)
    if val <= 0:
        raise ValueError("must be positive")

    # Market probability entered directly (e.g. 0.63)
    if 0 < val < 1:
        return val

    # Decimal odds (e.g. 1.67, 2.30)
    if 1.0 <= val <= 20.0:
        return 1.0 / val

    # Implied percent (e.g. 60)
    if 20.0 < val <= 100.0:
        return val / 100.0

    # Unsigned American positive odds (e.g. 150 means +150)
    if val > 100.0:
        return american_to_implied(f"+{int(round(val))}")

    raise ValueError("unsupported odds format")


def remove_vig(imp1, imp2):
    total = imp1 + imp2
    if total <= 0:
        return 0.5, 0.5
    return imp1 / total, imp2 / total


def kelly_fraction(prob_model, market_prob):
    if market_prob <= 0 or market_prob >= 1:
        return 0.0
    b = (1.0 / market_prob) - 1.0
    q = 1.0 - prob_model
    f = (b * prob_model - q) / b
    return max(0.0, f)


def expected_roi(prob_model, market_prob):
    # Buying a YES share priced at market_prob that pays 1 if winner occurs.
    return (prob_model / market_prob) - 1.0


def load_artifacts():
    try:
        model = joblib.load("xgb_model_v2.pkl")
        imputer = joblib.load("imputer_v2.pkl")
        arts = joblib.load("model_artifacts_v2.pkl")
    except FileNotFoundError:
        print("\n[ERROR] v2 artifacts not found.")
        print("Run: python ufc_model_v2.py")
        sys.exit(1)

    calibrator = None
    try:
        calibrator = joblib.load("isotonic_calibrator_v2.pkl")
    except FileNotFoundError:
        pass

    elo_wc = {}
    for raw_key, val in arts["elo_wc"].items():
        try:
            elo_wc[ast.literal_eval(raw_key)] = val
        except Exception:
            continue

    return model, imputer, calibrator, arts, elo_wc


def resolve_name(query, all_fighters, fight_hist, elo_global, elo_base):
    query = query.strip()
    if query in all_fighters:
        return query

    candidates = get_close_matches(query, all_fighters, n=5, cutoff=0.6)
    if not candidates:
        print(f"\n[!] Could not find '{query}'.")
        sys.exit(1)

    if len(candidates) == 1:
        print(f"[~] '{query}' matched to: {candidates[0]}")
        return candidates[0]

    print(f"\nMultiple matches for '{query}':")
    for i, c in enumerate(candidates, 1):
        fights = len(fight_hist.get(c, []))
        elo = elo_global.get(c, elo_base)
        print(f"  {i}. {c:<35} ELO={elo:.0f} ({fights} fights)")

    choice = input("Enter number: ").strip()
    try:
        return candidates[int(choice) - 1]
    except Exception:
        print("Invalid selection")
        sys.exit(1)


def build_features(f1, f2, wc, fight_date, arts, elo_wc):
    elo_global = arts["elo_global"]
    fight_hist = arts["fight_hist"]
    fighter_phys = arts["fighter_phys"]
    elo_base = arts["ELO_BASE"]
    wc_weight = arts["WC_WEIGHT"]
    mean_wc_lbs = arts["MEAN_WC_LBS"]

    f1_elo = elo_global.get(f1, elo_base)
    f2_elo = elo_global.get(f2, elo_base)
    f1_elo_wc = elo_wc.get((f1, wc), elo_base)
    f2_elo_wc = elo_wc.get((f2, wc), elo_base)

    h1 = fight_hist.get(f1, [])
    h2 = fight_hist.get(f2, [])

    s1_10 = _rolling(h1, 10, elo_base)
    s1_25 = _rolling(h1, 25, elo_base)
    s1_all = _rolling(h1, None, elo_base)
    s2_10 = _rolling(h2, 10, elo_base)
    s2_25 = _rolling(h2, 25, elo_base)
    s2_all = _rolling(h2, None, elo_base)

    f1_w_streak = _streak(h1, True)
    f1_l_streak = _streak(h1, False)
    f2_w_streak = _streak(h2, True)
    f2_l_streak = _streak(h2, False)

    h2h_f1w, h2h_f1l = _h2h(fight_hist, f1, f2)

    f1_last = h1[-1]["date"] if h1 else None
    f2_last = h2[-1]["date"] if h2 else None
    f1_rest = _safe((fight_date - f1_last).days) if f1_last else 365.0
    f2_rest = _safe((fight_date - f2_last).days) if f2_last else 365.0

    p1 = fighter_phys.get(f1, {})
    p2 = fighter_phys.get(f2, {})

    f1_h = p1.get("height_in", np.nan)
    f2_h = p2.get("height_in", np.nan)
    f1_r = p1.get("reach_in", np.nan)
    f2_r = p2.get("reach_in", np.nan)
    f1_dob = p1.get("dob", pd.NaT)
    f2_dob = p2.get("dob", pd.NaT)
    f1_stance = p1.get("stance") or "Unknown"
    f2_stance = p2.get("stance") or "Unknown"

    f1_age = _age_at(f1_dob, fight_date)
    f2_age = _age_at(f2_dob, fight_date)

    feat = {}
    feat["elo_diff"] = f1_elo - f2_elo
    feat["elo_wc_diff"] = f1_elo_wc - f2_elo_wc
    feat["f1_elo"] = f1_elo
    feat["f2_elo"] = f2_elo
    feat["sum_elo"] = f1_elo + f2_elo

    feat["wc_lbs"] = float(wc_weight.get(wc, mean_wc_lbs))

    feat["avg_opp_elo_diff_10"] = s1_10["avg_opp_elo"] - s2_10["avg_opp_elo"]
    feat["avg_opp_elo_diff_25"] = s1_25["avg_opp_elo"] - s2_25["avg_opp_elo"]
    feat["f1_avg_opp_elo_25"] = s1_25["avg_opp_elo"]
    feat["f2_avg_opp_elo_25"] = s2_25["avg_opp_elo"]

    feat["height_diff"] = _diff(f1_h, f2_h)
    feat["reach_diff"] = _diff(f1_r, f2_r)
    feat["age_diff"] = _diff(f1_age, f2_age)
    feat["f1_age"] = f1_age
    feat["f2_age"] = f2_age

    feat["f1_orthodox"] = int(f1_stance == "Orthodox")
    feat["f2_orthodox"] = int(f2_stance == "Orthodox")
    feat["f1_southpaw"] = int(f1_stance == "Southpaw")
    feat["f2_southpaw"] = int(f2_stance == "Southpaw")
    feat["ortho_vs_south"] = int(
        (f1_stance == "Orthodox" and f2_stance == "Southpaw")
        or (f1_stance == "Southpaw" and f2_stance == "Orthodox")
    )

    feat["wr_diff_10"] = s1_10["win_rate"] - s2_10["win_rate"]
    feat["wr_diff_25"] = s1_25["win_rate"] - s2_25["win_rate"]
    feat["wr_diff_all"] = s1_all["win_rate"] - s2_all["win_rate"]

    feat["ko_rate_diff"] = s1_25["ko_rate"] - s2_25["ko_rate"]
    feat["sub_rate_diff"] = s1_25["sub_rate"] - s2_25["sub_rate"]
    feat["dec_rate_diff"] = s1_25["dec_rate"] - s2_25["dec_rate"]

    feat["sig_diff"] = s1_25["avg_sig"] - s2_25["avg_sig"]
    feat["td_diff"] = s1_25["avg_td"] - s2_25["avg_td"]
    feat["ctrl_diff"] = s1_25["avg_ctrl"] - s2_25["avg_ctrl"]

    feat["exp_diff"] = s1_all["count"] - s2_all["count"]
    feat["f1_fights"] = s1_all["count"]
    feat["f2_fights"] = s2_all["count"]

    feat["win_streak_diff"] = f1_w_streak - f2_w_streak
    feat["lose_streak_diff"] = f1_l_streak - f2_l_streak
    feat["f1_win_streak"] = f1_w_streak
    feat["f2_win_streak"] = f2_w_streak

    feat["h2h_diff"] = h2h_f1w - h2h_f1l
    feat["h2h_met"] = int((h2h_f1w + h2h_f1l) > 0)

    feat["rest_diff"] = f1_rest - f2_rest
    feat["f1_rest_days"] = f1_rest
    feat["f2_rest_days"] = f2_rest

    info = {
        "f1_elo": f1_elo,
        "f2_elo": f2_elo,
        "f1_elo_wc": f1_elo_wc,
        "f2_elo_wc": f2_elo_wc,
        "f1_fights": s1_all["count"],
        "f2_fights": s2_all["count"],
        "f1_wr_25": s1_25["win_rate"],
        "f2_wr_25": s2_25["win_rate"],
        "f1_w_streak": f1_w_streak,
        "f2_w_streak": f2_w_streak,
        "f1_ko_rate": s1_25["ko_rate"],
        "f2_ko_rate": s2_25["ko_rate"],
        "f1_sub_rate": s1_25["sub_rate"],
        "f2_sub_rate": s2_25["sub_rate"],
        "f1_stance": f1_stance,
        "f2_stance": f2_stance,
        "f1_age": f1_age,
        "f2_age": f2_age,
    }
    return feat, info


def direction_probability(f1, f2, wc, fight_date, model, imputer, calibrator, arts, elo_wc):
    feat, info = build_features(f1, f2, wc, fight_date, arts, elo_wc)
    feature_cols = arts["FEATURE_COLS"]
    row = {col: feat.get(col, np.nan) for col in feature_cols}
    X = pd.DataFrame([row])[feature_cols]
    X_imp = imputer.transform(X)

    p_raw = float(model.predict_proba(X_imp)[0, 1])
    p_cal = float(calibrator.transform([p_raw])[0]) if calibrator is not None else p_raw
    return p_raw, p_cal, info


def predict_matchup(f1, f2, wc, fight_date, model, imputer, calibrator, arts, elo_wc, use_symmetry=True):
    p_ab_raw, p_ab_cal, info = direction_probability(
        f1, f2, wc, fight_date, model, imputer, calibrator, arts, elo_wc
    )

    if use_symmetry:
        p_ba_raw, p_ba_cal, _ = direction_probability(
            f2, f1, wc, fight_date, model, imputer, calibrator, arts, elo_wc
        )
        p_f1 = 0.5 * (p_ab_cal + (1.0 - p_ba_cal))
        consistency_gap = abs(p_ab_cal - (1.0 - p_ba_cal))
    else:
        p_f1 = p_ab_cal
        consistency_gap = 0.0

    p_f2 = 1.0 - p_f1
    winner = f1 if p_f1 >= 0.5 else f2
    win_prob = max(p_f1, p_f2)

    if win_prob >= 0.68:
        confidence = "HIGH"
    elif win_prob >= 0.58:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    print(f"\n{SEP}")
    print(f" PREDICTION (v2): {f1} vs {f2}")
    print(f" Division: {wc}")
    print(SEP)

    bar_len = 40
    f1_bar = round(p_f1 * bar_len)
    f2_bar = bar_len - f1_bar
    print(f"\n {f1:<25} {'|' * f1_bar}{'.' * f2_bar}  {f2}")
    print(f" {'':25} {p_f1*100:>5.1f}%  vs  {p_f2*100:.1f}%")

    print(f"\n Predicted winner : {winner}")
    print(f" Calibrated prob  : {win_prob*100:.1f}%")
    print(f" Confidence       : {confidence}")
    if use_symmetry:
        print(f" Symmetry gap     : {consistency_gap*100:.2f}% (lower is better)")

    print("\n FAIR VALUE (price model)")
    print(f" - Fair YES price on {f1}: {p_f1:.4f} ({p_f1*100:.2f}c)")
    print(f" - Fair YES price on {f2}: {p_f2:.4f} ({p_f2*100:.2f}c)")
    print(f" - Fair decimal odds: {f1}={1/max(p_f1,1e-9):.3f}, {f2}={1/max(p_f2,1e-9):.3f}")

    print(f"\n {'STAT':<26} {f1[:18]:<20} {f2[:18]}")
    print(f" {'-'*64}")
    rows = [
        ("Overall ELO", f"{info['f1_elo']:.0f}", f"{info['f2_elo']:.0f}"),
        (f"{wc} ELO", f"{info['f1_elo_wc']:.0f}", f"{info['f2_elo_wc']:.0f}"),
        ("UFC fights", f"{info['f1_fights']}", f"{info['f2_fights']}"),
        ("Win rate (last 25)", f"{info['f1_wr_25']*100:.0f}%", f"{info['f2_wr_25']*100:.0f}%"),
        ("Win streak", f"{info['f1_w_streak']}", f"{info['f2_w_streak']}"),
        ("KO rate (last 25)", f"{info['f1_ko_rate']*100:.0f}%", f"{info['f2_ko_rate']*100:.0f}%"),
        ("Sub rate (last 25)", f"{info['f1_sub_rate']*100:.0f}%", f"{info['f2_sub_rate']*100:.0f}%"),
        ("Stance", info["f1_stance"], info["f2_stance"]),
        (
            "Age",
            f"{info['f1_age']:.1f}" if not np.isnan(info["f1_age"]) else "N/A",
            f"{info['f2_age']:.1f}" if not np.isnan(info["f2_age"]) else "N/A",
        ),
    ]
    for stat, v1, v2 in rows:
        print(f" {stat:<26} {v1:<20} {v2}")

    return p_f1, p_f2, confidence


def betting_analysis(f1, f2, p_f1, p_f2):
    print(f"\n{SEP}")
    print(" BETTING ANALYSIS (v2)")
    print(SEP)
    print(
        "\nEnter market odds for each fighter. Supported:\n"
        "- American: -150 or +130\n"
        "- Decimal odds: 1.67 or 2.30\n"
        "- Implied probability: 0.62 or 62 or 62%\n"
    )

    def ask_prob(name):
        while True:
            raw = input(f" Odds / implied prob for {name}: ").strip()
            try:
                return parse_market_probability(raw)
            except Exception as e:
                print(f"  Invalid format ({e}). Try again.")

    raw_imp_f1 = ask_prob(f1)
    raw_imp_f2 = ask_prob(f2)

    mkt_f1, mkt_f2 = remove_vig(raw_imp_f1, raw_imp_f2)

    edge_f1 = p_f1 - mkt_f1
    edge_f2 = p_f2 - mkt_f2

    roi_f1 = expected_roi(p_f1, mkt_f1)
    roi_f2 = expected_roi(p_f2, mkt_f2)

    kelly_f1 = kelly_fraction(p_f1, mkt_f1)
    kelly_f2 = kelly_fraction(p_f2, mkt_f2)
    half_kelly_f1 = 0.5 * kelly_f1
    half_kelly_f2 = 0.5 * kelly_f2

    print(f"\n {'':28} {f1[:20]:<22} {f2[:20]}")
    print(f" {'-'*72}")
    print(f" {'Raw market implied':28} {raw_imp_f1*100:6.2f}%{'':<12} {raw_imp_f2*100:6.2f}%")
    print(f" {'Vig-removed market':28} {mkt_f1*100:6.2f}%{'':<12} {mkt_f2*100:6.2f}%")
    print(f" {'Model fair prob':28} {p_f1*100:6.2f}%{'':<12} {p_f2*100:6.2f}%")
    print(f" {'Edge (model-market)':28} {edge_f1*100:+6.2f}%{'':<11} {edge_f2*100:+6.2f}%")
    print(f" {'Expected ROI per $1':28} {roi_f1*100:+6.2f}%{'':<11} {roi_f2*100:+6.2f}%")
    print(f" {'Half Kelly stake':28} {half_kelly_f1*100:6.2f}%{'':<12} {half_kelly_f2*100:6.2f}%")

    edge_threshold = 0.05
    min_conf = 0.55

    best_edge = max(edge_f1, edge_f2)
    if edge_f1 >= edge_f2:
        best_name, best_prob, best_roi, best_kelly = f1, p_f1, roi_f1, half_kelly_f1
    else:
        best_name, best_prob, best_roi, best_kelly = f2, p_f2, roi_f2, half_kelly_f2

    print(f"\n{SEP}")
    print(" RECOMMENDATION")
    print(SEP)

    if best_edge < edge_threshold:
        print(f"PASS - edge {best_edge*100:+.2f}% is below {edge_threshold*100:.1f}% threshold.")
    elif best_prob < min_conf:
        print(f"PASS - confidence {best_prob*100:.1f}% is below {min_conf*100:.1f}%.")
    else:
        print(f"BET {best_name.upper()}")
        print(f" Edge       : {best_edge*100:+.2f}%")
        print(f" Exp ROI    : {best_roi*100:+.2f}% per $1")
        print(f" Half Kelly : {best_kelly*100:.2f}% bankroll")


def main():
    parser = argparse.ArgumentParser(description="UFC predictor v2")
    parser.add_argument("fighter1", nargs="?", help="Fighter 1")
    parser.add_argument("fighter2", nargs="?", help="Fighter 2")
    parser.add_argument("--wc", default=None, help="Weight class")
    parser.add_argument("--date", default=None, help="Fight date YYYY-MM-DD (optional)")
    parser.add_argument("--no-symmetry", action="store_true", help="Disable A/B and B/A averaging")
    args = parser.parse_args()

    model, imputer, calibrator, arts, elo_wc = load_artifacts()
    elo_global = arts["elo_global"]
    fighter_phys = arts["fighter_phys"]
    all_fighters = sorted({n for n in (list(elo_global.keys()) + list(fighter_phys.keys())) if isinstance(n, str) and n.strip()})

    print(f"\n{SEP}")
    print(" UFC FIGHT PREDICTOR (v2)")
    print(SEP)
    print(f" fighters={len(all_fighters):,} | build_mode={arts.get('build_mode', 'unknown')}")
    print(f" calibrated={'yes' if calibrator is not None else 'no'}")

    f1_raw = args.fighter1 if args.fighter1 else input("\nFighter 1 name: ").strip()
    f2_raw = args.fighter2 if args.fighter2 else input("Fighter 2 name: ").strip()

    f1 = resolve_name(f1_raw, all_fighters, arts["fight_hist"], elo_global, arts["ELO_BASE"])
    f2 = resolve_name(f2_raw, all_fighters, arts["fight_hist"], elo_global, arts["ELO_BASE"])

    known_wcs = sorted(arts["WC_WEIGHT"].keys())
    if args.wc:
        wc = args.wc
    else:
        print(f"\nKnown weight classes: {', '.join(known_wcs)}")
        wc_in = input("Weight class (Enter for Lightweight): ").strip()
        if not wc_in:
            wc = "Lightweight"
        else:
            match = get_close_matches(wc_in, known_wcs, n=1, cutoff=0.5)
            wc = match[0] if match else wc_in

    if args.date:
        try:
            fight_date = pd.Timestamp(args.date)
        except Exception:
            print("Invalid --date format; expected YYYY-MM-DD")
            sys.exit(1)
    else:
        fight_date = pd.Timestamp(datetime.today().date())

    p_f1, p_f2, confidence = predict_matchup(
        f1,
        f2,
        wc,
        fight_date,
        model,
        imputer,
        calibrator,
        arts,
        elo_wc,
        use_symmetry=(not args.no_symmetry),
    )

    if confidence != "LOW":
        do_bet = input("\nDo you want odds/EV analysis? (y/n): ").strip().lower()
        if do_bet == "y":
            betting_analysis(f1, f2, p_f1, p_f2)
    else:
        print("\nSkipping betting analysis due to low-confidence prediction.")


if __name__ == "__main__":
    main()
