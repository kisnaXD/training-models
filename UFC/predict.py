#!/usr/bin/env python3
"""
UFC Fight Predictor
===================
Interactive tool that loads the trained model and lets you predict
the outcome of any UFC matchup.

Usage:
    python predict.py                  # interactive prompts
    python predict.py "Jon Jones" "Stipe Miocic" --wc "Heavyweight"

Outputs:
    - Model's win probability for each fighter
    - Confidence tier (HIGH / MEDIUM / LOW)
    - If you supply market odds: edge calculation + Kelly bet size
    - Plain-English betting recommendation
"""

import sys
import argparse
import joblib
import pandas as pd
import numpy as np
from difflib import get_close_matches
from datetime import datetime

# ── Load artifacts ────────────────────────────────────────────────
try:
    xgb_model  = joblib.load("xgb_model.pkl")
    imputer    = joblib.load("imputer.pkl")
    arts       = joblib.load("model_artifacts.pkl")
except FileNotFoundError:
    print("\n[ERROR] Model files not found.")
    print("        Run  python ufc_model.py  first to train and save the model.")
    sys.exit(1)

elo_global   = arts["elo_global"]
elo_wc_raw   = arts["elo_wc"]          # keys are stringified tuples "(name, wc)"
fight_hist   = arts["fight_hist"]
fighter_phys = arts["fighter_phys"]
FEATURE_COLS = arts["FEATURE_COLS"]
WC_WEIGHT    = arts["WC_WEIGHT"]
MEAN_WC_LBS  = arts["MEAN_WC_LBS"]
ELO_BASE     = arts["ELO_BASE"]

# Rebuild elo_wc with proper tuple keys
elo_wc = {}
for raw_key, val in elo_wc_raw.items():
    # key was saved as str(tuple), e.g. "('Jon Jones', 'Light Heavyweight')"
    try:
        import ast
        tup = ast.literal_eval(raw_key)
        elo_wc[tup] = val
    except Exception:
        pass

ALL_FIGHTERS = sorted(set(list(elo_global.keys()) + list(fighter_phys.keys())))
SEP = "=" * 65


# ── Helper functions (mirrors ufc_model.py) ───────────────────────

def _safe(val):
    try:
        v = float(val)
        return 0.0 if np.isnan(v) else v
    except Exception:
        return 0.0


def _age_at(dob, date):
    if pd.isna(dob) or date is None:
        return np.nan
    return (date - dob).days / 365.25


def _rolling(hist, n=None):
    h = hist[-n:] if n else hist
    total = len(h)
    if total == 0:
        return dict(
            win_rate=0.5, ko_rate=0.0, sub_rate=0.0, dec_rate=0.0,
            avg_sig=0.0, avg_td=0.0, avg_ctrl=0.0, count=0,
            avg_opp_elo=float(ELO_BASE),
        )
    wins = sum(x["w"] for x in h)
    return dict(
        win_rate    = wins / total,
        ko_rate     = sum(x["w"] and x["ko"]  for x in h) / total,
        sub_rate    = sum(x["w"] and x["sub"] for x in h) / total,
        dec_rate    = sum(x["w"] and x["dec"] for x in h) / total,
        avg_sig     = float(np.mean([x["sig"]     for x in h])),
        avg_td      = float(np.mean([x["td"]      for x in h])),
        avg_ctrl    = float(np.mean([x["ctrl"]    for x in h])),
        count       = total,
        avg_opp_elo = float(np.mean([x.get("opp_elo", ELO_BASE) for x in h])),
    )


def _streak(hist, won_val):
    streak = 0
    for x in reversed(hist):
        if bool(x["w"]) == won_val:
            streak += 1
        else:
            break
    return streak


def _h2h(me, opp):
    h = fight_hist.get(me, [])
    wins   = sum(1 for x in h if x["opp"] == opp and     x["w"])
    losses = sum(1 for x in h if x["opp"] == opp and not x["w"])
    return wins, losses


def _diff(a, b):
    if np.isnan(a) or np.isnan(b):
        return np.nan
    return a - b


# ── Name fuzzy-matching ───────────────────────────────────────────

def resolve_name(query: str) -> str:
    """Return the closest known fighter name, or raise if ambiguous."""
    query = query.strip()
    if query in elo_global or query in fighter_phys:
        return query

    candidates = get_close_matches(query, ALL_FIGHTERS, n=5, cutoff=0.6)
    if not candidates:
        print(f"\n  [!] Could not find '{query}' in fighter database.")
        print(f"      Try a different spelling or check the name.")
        sys.exit(1)

    if len(candidates) == 1:
        print(f"  [~] '{query}' matched to: {candidates[0]}")
        return candidates[0]

    print(f"\n  Multiple matches for '{query}':")
    for i, c in enumerate(candidates, 1):
        fights = len(fight_hist.get(c, []))
        elo    = elo_global.get(c, ELO_BASE)
        print(f"    {i}. {c:<35}  ELO: {elo:.0f}  ({fights} UFC fights)")
    choice = input("  Enter number: ").strip()
    try:
        return candidates[int(choice) - 1]
    except (ValueError, IndexError):
        print("  Invalid choice.")
        sys.exit(1)


# ── Feature builder for a single matchup ─────────────────────────

def build_features(f1: str, f2: str, wc: str, fight_date=None):
    """Compute the full 41-feature vector for f1 vs f2."""
    if fight_date is None:
        fight_date = pd.Timestamp(datetime.today().date())

    f1_elo    = elo_global.get(f1, ELO_BASE)
    f2_elo    = elo_global.get(f2, ELO_BASE)
    f1_elo_wc = elo_wc.get((f1, wc), ELO_BASE)
    f2_elo_wc = elo_wc.get((f2, wc), ELO_BASE)

    h1 = fight_hist.get(f1, [])
    h2 = fight_hist.get(f2, [])

    s1_10  = _rolling(h1, 10)
    s1_25  = _rolling(h1, 25)
    s1_all = _rolling(h1)
    s2_10  = _rolling(h2, 10)
    s2_25  = _rolling(h2, 25)
    s2_all = _rolling(h2)

    f1_w_streak = _streak(h1, True)
    f1_l_streak = _streak(h1, False)
    f2_w_streak = _streak(h2, True)
    f2_l_streak = _streak(h2, False)

    h2h_f1w, h2h_f1l = _h2h(f1, f2)

    f1_last = h1[-1]["date"] if h1 else None
    f2_last = h2[-1]["date"] if h2 else None
    f1_rest = _safe((fight_date - f1_last).days) if f1_last else 365.0
    f2_rest = _safe((fight_date - f2_last).days) if f2_last else 365.0

    p1 = fighter_phys.get(f1, {})
    p2 = fighter_phys.get(f2, {})

    f1_h      = p1.get("height_in",  np.nan)
    f2_h      = p2.get("height_in",  np.nan)
    f1_r      = p1.get("reach_in",   np.nan)
    f2_r      = p2.get("reach_in",   np.nan)
    f1_dob    = p1.get("dob",        pd.NaT)
    f2_dob    = p2.get("dob",        pd.NaT)
    f1_stance = (p1.get("stance") or "Unknown")
    f2_stance = (p2.get("stance") or "Unknown")
    f1_age    = _age_at(f1_dob, fight_date)
    f2_age    = _age_at(f2_dob, fight_date)

    feat = {}
    feat["elo_diff"]    = f1_elo    - f2_elo
    feat["elo_wc_diff"] = f1_elo_wc - f2_elo_wc
    feat["f1_elo"]      = f1_elo
    feat["f2_elo"]      = f2_elo
    feat["sum_elo"]     = f1_elo + f2_elo

    feat["wc_lbs"] = float(WC_WEIGHT.get(wc, MEAN_WC_LBS))

    feat["avg_opp_elo_diff_10"] = s1_10["avg_opp_elo"] - s2_10["avg_opp_elo"]
    feat["avg_opp_elo_diff_25"] = s1_25["avg_opp_elo"] - s2_25["avg_opp_elo"]
    feat["f1_avg_opp_elo_25"]   = s1_25["avg_opp_elo"]
    feat["f2_avg_opp_elo_25"]   = s2_25["avg_opp_elo"]

    feat["height_diff"] = _diff(f1_h,   f2_h)
    feat["reach_diff"]  = _diff(f1_r,   f2_r)
    feat["age_diff"]    = _diff(f1_age, f2_age)
    feat["f1_age"]      = f1_age
    feat["f2_age"]      = f2_age

    feat["f1_orthodox"]    = int(f1_stance == "Orthodox")
    feat["f2_orthodox"]    = int(f2_stance == "Orthodox")
    feat["f1_southpaw"]    = int(f1_stance == "Southpaw")
    feat["f2_southpaw"]    = int(f2_stance == "Southpaw")
    feat["ortho_vs_south"] = int(
        (f1_stance == "Orthodox" and f2_stance == "Southpaw") or
        (f1_stance == "Southpaw" and f2_stance == "Orthodox")
    )

    feat["wr_diff_10"]  = s1_10["win_rate"]  - s2_10["win_rate"]
    feat["wr_diff_25"]  = s1_25["win_rate"]  - s2_25["win_rate"]
    feat["wr_diff_all"] = s1_all["win_rate"] - s2_all["win_rate"]

    feat["ko_rate_diff"]  = s1_25["ko_rate"]  - s2_25["ko_rate"]
    feat["sub_rate_diff"] = s1_25["sub_rate"] - s2_25["sub_rate"]
    feat["dec_rate_diff"] = s1_25["dec_rate"] - s2_25["dec_rate"]

    feat["sig_diff"]  = s1_25["avg_sig"]  - s2_25["avg_sig"]
    feat["td_diff"]   = s1_25["avg_td"]   - s2_25["avg_td"]
    feat["ctrl_diff"] = s1_25["avg_ctrl"] - s2_25["avg_ctrl"]

    feat["exp_diff"]  = s1_all["count"] - s2_all["count"]
    feat["f1_fights"] = s1_all["count"]
    feat["f2_fights"] = s2_all["count"]

    feat["win_streak_diff"]  = f1_w_streak - f2_w_streak
    feat["lose_streak_diff"] = f1_l_streak - f2_l_streak
    feat["f1_win_streak"]    = f1_w_streak
    feat["f2_win_streak"]    = f2_w_streak

    feat["h2h_diff"] = h2h_f1w - h2h_f1l
    feat["h2h_met"]  = int((h2h_f1w + h2h_f1l) > 0)

    feat["rest_diff"]     = f1_rest - f2_rest
    feat["f1_rest_days"]  = f1_rest
    feat["f2_rest_days"]  = f2_rest

    return feat, {
        "f1_elo": f1_elo, "f2_elo": f2_elo,
        "f1_elo_wc": f1_elo_wc, "f2_elo_wc": f2_elo_wc,
        "f1_fights": s1_all["count"], "f2_fights": s2_all["count"],
        "f1_wr_25": s1_25["win_rate"], "f2_wr_25": s2_25["win_rate"],
        "f1_w_streak": f1_w_streak, "f2_w_streak": f2_w_streak,
        "f1_ko_rate": s1_25["ko_rate"], "f2_ko_rate": s2_25["ko_rate"],
        "f1_sub_rate": s1_25["sub_rate"], "f2_sub_rate": s2_25["sub_rate"],
        "f1_stance": f1_stance, "f2_stance": f2_stance,
        "f1_age": f1_age, "f2_age": f2_age,
        "h2h_f1w": h2h_f1w, "h2h_f1l": h2h_f1l,
    }


# ── Prediction & display ──────────────────────────────────────────

def predict(f1: str, f2: str, wc: str, fight_date=None):
    feat, info = build_features(f1, f2, wc, fight_date)

    # Build feature row in the exact column order the model was trained on
    row = {col: feat.get(col, np.nan) for col in FEATURE_COLS}
    X   = pd.DataFrame([row])[FEATURE_COLS]
    X_imp = imputer.transform(X)

    prob_f1 = float(xgb_model.predict_proba(X_imp)[0, 1])
    prob_f2 = 1.0 - prob_f1

    winner  = f1 if prob_f1 >= 0.5 else f2
    win_prob = max(prob_f1, prob_f2)

    if win_prob >= 0.68:
        confidence = "HIGH"
        conf_note  = "Model has strong conviction."
    elif win_prob >= 0.58:
        confidence = "MEDIUM"
        conf_note  = "Reasonable lean. Don't bet big."
    else:
        confidence = "LOW"
        conf_note  = "Near coin-flip. Avoid."

    # ── Print results ─────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  PREDICTION:  {f1}  vs  {f2}")
    print(f"  Division:    {wc}")
    print(SEP)

    bar_len = 40
    f1_bar = round(prob_f1 * bar_len)
    f2_bar = bar_len - f1_bar
    print(f"\n  {f1:<25} {'|' * f1_bar}{'.' * f2_bar}  {f2}")
    print(f"  {'':25} {prob_f1*100:>5.1f}%  vs  {prob_f2*100:.1f}%")

    print(f"\n  Predicted winner : {winner}")
    print(f"  Model probability: {win_prob*100:.1f}%")
    print(f"  Confidence       : {confidence}  — {conf_note}")

    # ── Fighter stats side-by-side ────────────────────────────────
    print(f"\n  {'STAT':<28} {f1[:18]:<20} {f2[:18]}")
    print(f"  {'-'*65}")
    rows = [
        ("Overall ELO",       f"{info['f1_elo']:.0f}",         f"{info['f2_elo']:.0f}"),
        (f"{wc} ELO",         f"{info['f1_elo_wc']:.0f}",      f"{info['f2_elo_wc']:.0f}"),
        ("UFC fights",        f"{info['f1_fights']}",           f"{info['f2_fights']}"),
        ("Win rate (last 25)",f"{info['f1_wr_25']*100:.0f}%",  f"{info['f2_wr_25']*100:.0f}%"),
        ("Win streak",        f"{info['f1_w_streak']}",         f"{info['f2_w_streak']}"),
        ("KO rate (last 25)", f"{info['f1_ko_rate']*100:.0f}%",f"{info['f2_ko_rate']*100:.0f}%"),
        ("Sub rate (last 25)",f"{info['f1_sub_rate']*100:.0f}%",f"{info['f2_sub_rate']*100:.0f}%"),
        ("Stance",            info["f1_stance"],                 info["f2_stance"]),
        ("Age",               f"{info['f1_age']:.1f}" if not np.isnan(info['f1_age']) else "N/A",
                              f"{info['f2_age']:.1f}" if not np.isnan(info['f2_age']) else "N/A"),
    ]
    if info["h2h_f1w"] + info["h2h_f1l"] > 0:
        rows.append(("Head-to-head",
                     f"{f1[:10]} {info['h2h_f1w']}-{info['h2h_f1l']} {f2[:10]}", ""))
    for stat, v1, v2 in rows:
        print(f"  {stat:<28} {v1:<20} {v2}")

    return prob_f1, prob_f2, confidence


# ── Betting edge calculator ───────────────────────────────────────

def american_to_implied(odds: str) -> float:
    """Convert American odds string (e.g. '-150' or '+130') to implied probability."""
    o = float(odds.replace(" ", ""))
    if o < 0:
        return abs(o) / (abs(o) + 100)
    else:
        return 100 / (o + 100)


def remove_vig(imp1: float, imp2: float):
    """Strip the vig from two implied probabilities so they sum to 1.0."""
    total = imp1 + imp2
    return imp1 / total, imp2 / total


def kelly_fraction(prob_model: float, prob_implied: float) -> float:
    """
    Full Kelly: f = (b*p - q) / b
    where b = net odds on a $1 bet = (1/implied - 1)
          p = model probability
          q = 1 - p
    Returns fraction of bankroll to bet (0 if no edge).
    """
    if prob_implied >= 1.0 or prob_implied <= 0.0:
        return 0.0
    b = (1.0 / prob_implied) - 1.0
    p = prob_model
    q = 1.0 - p
    f = (b * p - q) / b
    return max(0.0, f)


def betting_analysis(f1: str, f2: str, prob_f1: float, prob_f2: float):
    print(f"\n{SEP}")
    print("  BETTING ANALYSIS")
    print(SEP)
    print("""
  Enter the market odds for each fighter.
  Accepted formats:
    American:  -150   or   +130   (most US sportsbooks)
    Decimal:   1.67   or   2.30   (bet365, Pinnacle, Polymarket)
    Implied %: 60     (just type the percentage, e.g. 60 for 60%)
""")

    def get_implied(fighter_name: str) -> float:
        raw = input(f"  Odds / implied % for {fighter_name}: ").strip()
        try:
            val = float(raw)
            # Distinguish: implied % vs American vs decimal
            if 0 < val < 1:          # decimal like 0.65 -> already a probability
                return val
            if 1 <= val <= 100:       # percentage entry like "60"
                return val / 100.0
            if val > 100:             # decimal odds like 1.67
                return 1.0 / val
            if val < 0:               # American negative like -150
                return american_to_implied(raw)
        except ValueError:
            pass
        # fallback: treat as American string
        return american_to_implied(raw)

    raw_imp_f1 = get_implied(f1)
    raw_imp_f2 = get_implied(f2)

    # Strip vig so they sum to exactly 1.0
    true_imp_f1, true_imp_f2 = remove_vig(raw_imp_f1, raw_imp_f2)

    edge_f1 = prob_f1 - true_imp_f1
    edge_f2 = prob_f2 - true_imp_f2

    kelly_f1 = kelly_fraction(prob_f1, true_imp_f1)
    kelly_f2 = kelly_fraction(prob_f2, true_imp_f2)

    # Half Kelly is recommended in practice (full Kelly is very aggressive)
    half_kelly_f1 = kelly_f1 / 2
    half_kelly_f2 = kelly_f2 / 2

    print(f"\n  {'':30} {f1[:20]:<22} {f2[:20]}")
    print(f"  {'-'*72}")
    print(f"  {'Raw market implied':30} {raw_imp_f1*100:.1f}%{'':<17} {raw_imp_f2*100:.1f}%")
    print(f"  {'Market implied (vig removed)':30} {true_imp_f1*100:.1f}%{'':<17} {true_imp_f2*100:.1f}%")
    print(f"  {'Model probability':30} {prob_f1*100:.1f}%{'':<17} {prob_f2*100:.1f}%")
    print(f"  {'Edge (model - market)':30} {edge_f1*100:+.1f}%{'':<16} {edge_f2*100:+.1f}%")
    print(f"  {'Kelly fraction (full)':30} {kelly_f1*100:.1f}%{'':<17} {kelly_f2*100:.1f}%")
    print(f"  {'Kelly fraction (half — recommended)':30} {half_kelly_f1*100:.1f}%{'':<17} {half_kelly_f2*100:.1f}%")

    # ── Decision logic ────────────────────────────────────────────
    EDGE_THRESHOLD = 0.05  # minimum 5% edge to consider a bet
    MIN_CONFIDENCE_FOR_BET = 0.55  # model must give at least 55% to the selection

    print(f"\n{SEP}")
    print("  RECOMMENDATION")
    print(SEP)

    best_edge    = max(edge_f1, edge_f2)
    best_fighter = f1 if edge_f1 >= edge_f2 else f2
    best_prob    = prob_f1 if edge_f1 >= edge_f2 else prob_f2
    best_kelly   = half_kelly_f1 if edge_f1 >= edge_f2 else half_kelly_f2
    best_edge_v  = edge_f1 if edge_f1 >= edge_f2 else edge_f2

    if best_edge < EDGE_THRESHOLD:
        print(f"""
  PASS — edge too small to act on.

  Model edge over market: {best_edge*100:+.1f}%  (threshold: +5.0%)

  The market and model largely agree here. After accounting for
  the sportsbook's cut, there is insufficient edge to justify a bet.
  The expected value is near zero or negative.

  What to do: Skip this fight, or wait to see if the line moves.
""")
    elif best_prob < MIN_CONFIDENCE_FOR_BET:
        print(f"""
  PASS — model conviction too low.

  Model probability: {best_prob*100:.1f}%  (minimum: 55%)

  Even though there may be technical edge, the model is essentially
  saying this fight is a coin flip. Betting on coin flips is a
  bad long-run strategy regardless of the line.
""")
    else:
        print(f"""
  BET {best_fighter.upper()}

  Edge over market  : {best_edge_v*100:+.1f}%
  Model probability : {best_prob*100:.1f}%
  Half-Kelly stake  : {best_kelly*100:.1f}% of bankroll

  Example: If your bankroll is $1,000, bet ${best_kelly*1000:.0f} on {best_fighter}.

  Important: Half-Kelly is already conservative. Never bet more than
  the full Kelly fraction ({best_kelly*2*100:.1f}% here) — that path leads to ruin
  even when you have genuine edge.
""")

    # ── Plain English summary of the Luana/Melissa scenario ──────
    print(f"""
  HOW TO READ THIS (the Luana/Melissa 53-47 example):
  ─────────────────────────────────────────────────────
  Market says 53% Luana / 47% Melissa.
  After vig removal, real market probabilities are ~50.5% / 49.5%.
  That's essentially a coin flip to the market.

  IF the model says Luana at 62%:
    Edge = 62% - 50.5% = +11.5%  →  BET LUANA  (strong edge)

  IF the model says Luana at 54%:
    Edge = 54% - 50.5% = +3.5%   →  PASS  (edge below 5% threshold)

  IF the model says Luana at 44%:
    Edge on Melissa = 50.5% - 44% = ... you'd consider MELISSA
    even though the market leans slightly to Luana.

  The market leaning 53-47 tells you almost nothing actionable.
  What matters is how far the MODEL'S probability deviates from
  the market's vig-removed probability.
""")


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="UFC Fight Predictor")
    parser.add_argument("fighter1", nargs="?", help="Fighter 1 name")
    parser.add_argument("fighter2", nargs="?", help="Fighter 2 name")
    parser.add_argument("--wc",     default=None, help="Weight class (e.g. 'Lightweight')")
    args = parser.parse_args()

    print(f"\n{SEP}")
    print("  UFC FIGHT PREDICTOR")
    print(f"{SEP}")
    print(f"  {len(ALL_FIGHTERS):,} fighters in database | "
          f"Model trained through 2022, tested on 2024-2026")

    # Fighter names
    if args.fighter1:
        f1_raw = args.fighter1
    else:
        f1_raw = input("\n  Fighter 1 name: ").strip()
    if args.fighter2:
        f2_raw = args.fighter2
    else:
        f2_raw = input("  Fighter 2 name: ").strip()

    f1 = resolve_name(f1_raw)
    f2 = resolve_name(f2_raw)

    # Weight class
    known_wcs = sorted(WC_WEIGHT.keys())
    if args.wc:
        wc = args.wc
    else:
        print(f"\n  Known weight classes: {', '.join(known_wcs)}")
        wc_input = input("  Weight class (press Enter to auto-detect): ").strip()
        if not wc_input:
            # Try to infer from most recent fight
            h1 = fight_hist.get(f1, [])
            wc = "Unknown"
            if h1:
                # Last fight's division — not stored directly, use global ELO as fallback
                print("  [Auto] Weight class unknown — using 'Lightweight' as default.")
                wc = "Lightweight"
            else:
                wc = "Lightweight"
        else:
            # Fuzzy match weight class
            matches = get_close_matches(wc_input, known_wcs, n=1, cutoff=0.5)
            wc = matches[0] if matches else wc_input
            print(f"  Using weight class: {wc}")

    # Run prediction
    prob_f1, prob_f2, confidence = predict(f1, f2, wc)

    # Offer betting analysis
    if confidence != "LOW":
        do_bet = input(f"\n  Do you have market odds to analyse? (y/n): ").strip().lower()
        if do_bet == "y":
            betting_analysis(f1, f2, prob_f1, prob_f2)
    else:
        print("\n  [Skipping betting analysis — model confidence too low to be actionable.]\n")


if __name__ == "__main__":
    main()
