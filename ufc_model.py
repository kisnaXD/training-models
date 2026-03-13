#!/usr/bin/env python3
"""
UFC Fight Outcome Prediction Model
===================================
Inspired by the tennis ML approach (theGreenCoding / Australian Open article):

  - Custom ELO rating system: overall + weight-class-specific
    (analogous to the article's overall + surface-specific ELO for tennis)
  - Rolling historical stats at multiple windows (last 10 / 25 / 50 fights)
  - Physical differentials: height, reach, age, stance matchup
  - Head-to-head records, win/loss streaks, rest days

  Four algorithms tested for robustness:
    1. Decision Tree
    2. Random Forest
    3. XGBoost   ← expected best (same finding as article)
    4. Neural Network (MLP)

  Train / Validate / Test split is STRICTLY CHRONOLOGICAL — no data from
  the future is ever used to build features for earlier fights.
"""

import os
import pandas as pd
import numpy as np
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    classification_report, confusion_matrix,
)
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)
os.makedirs("plots", exist_ok=True)

SEP = "=" * 65


# ==================================================================
# SECTION 1 -- LOAD DATA
# ==================================================================

print(SEP)
print("  UFC FIGHT OUTCOME PREDICTION MODEL")
print(SEP)

fights_raw   = pd.read_csv("data/fights.csv")
fighters_raw = pd.read_csv("data/fighters.csv")

print(f"\n[DATA] {len(fights_raw):,} fights loaded")
print(f"[DATA] {len(fighters_raw):,} fighters loaded")

# Parse dates
fights_raw["event_date"]  = pd.to_datetime(fights_raw["event_date"],  errors="coerce")
fighters_raw["dob"]       = pd.to_datetime(fighters_raw["dob"],       errors="coerce")

# Drop rows missing the essentials
fights_raw = fights_raw.dropna(
    subset=["winner", "fighter1_name", "fighter2_name", "event_date"]
)
fights_raw = fights_raw.sort_values("event_date").reset_index(drop=True)

print(f"[DATA] {len(fights_raw):,} usable fights after dropping nulls")
print(
    f"[DATA] Date range: "
    f"{fights_raw['event_date'].min().date()} -> "
    f"{fights_raw['event_date'].max().date()}"
)


# ==================================================================
# SECTION 2 -- PARSE PHYSICAL ATTRIBUTES FROM fighters.csv
# ==================================================================

def _height_in(h):
    """Convert '5\' 11"' -> inches (float)."""
    try:
        if pd.isna(h):
            return np.nan
        h = str(h).strip()
        for ch in ["\u2019", "\u2018", "\u02bc"]:
            h = h.replace(ch, "'")
        for ch in ["\u201c", "\u201d", "\u2033", '"']:
            h = h.replace(ch, "")
        h = h.replace("'", "' ").strip()
        parts = h.split("'")
        feet   = int(parts[0].strip())
        inches = int(parts[1].strip()) if len(parts) > 1 and parts[1].strip().isdigit() else 0
        return feet * 12 + inches
    except Exception:
        return np.nan


def _reach_in(r):
    try:
        if pd.isna(r):
            return np.nan
        return float(str(r).replace('"', "").replace("\u2033", "").strip())
    except Exception:
        return np.nan


def _weight_lbs(w):
    try:
        if pd.isna(w):
            return np.nan
        return float(str(w).replace("lbs.", "").strip())
    except Exception:
        return np.nan


fighters_raw["height_in"]  = fighters_raw["height"].apply(_height_in)
fighters_raw["reach_in"]   = fighters_raw["reach"].apply(_reach_in)
fighters_raw["weight_lbs"] = fighters_raw["weight"].apply(_weight_lbs)

fighter_phys = (
    fighters_raw
    .drop_duplicates(subset="fighter_name", keep="last")
    .set_index("fighter_name")[["height_in", "reach_in", "weight_lbs", "dob", "stance"]]
    .to_dict("index")
)

print(f"\n[PHYS] Physical lookup built for {len(fighter_phys):,} fighters")
print(f"[PHYS] Height parsed: {fighters_raw['height_in'].notna().sum():,}  "
      f"| Reach parsed: {fighters_raw['reach_in'].notna().sum():,}")


# ==================================================================
# SECTION 3 -- ELO RATING SYSTEM
# ==================================================================
#
#  Identical mechanic to the tennis model:
#    - Every fighter starts at ELO_BASE = 1500
#    - K is now DYNAMIC (improvement #3 + #4):
#        KO/TKO finish  -> K=48  (most decisive outcome)
#        Submission     -> K=40
#        Decision       -> K=24  (least decisive)
#        Provisional (< 5 fights) -> K *= 2.0  (fast calibration)
#        Provisional (5-10 fights) -> K *= 1.5
#    - Gain/loss is proportional to rating gap — beating a stronger
#      opponent earns more points than beating a weaker one.
#
#  We compute TWO ELO tracks:
#    (a) Overall ELO  — across all weight classes
#    (b) Weight-class ELO — within each division
#      (mirrors the article's surface-specific ELO for clay/grass/hard)

ELO_BASE = 1500

# Improvement #1: Weight-class -> lbs mapping
# Gives the model an ordinal sense of the division (heavier = bigger/stronger)
WC_WEIGHT = {
    "Strawweight": 115,          "Women's Strawweight": 115,
    "Flyweight": 125,            "Women's Flyweight": 125,
    "Bantamweight": 135,         "Women's Bantamweight": 135,
    "Featherweight": 145,        "Women's Featherweight": 145,
    "Lightweight": 155,
    "Welterweight": 170,
    "Middleweight": 185,
    "Light Heavyweight": 205,
    "Heavyweight": 265,
    "Super Heavyweight": 265,
    "Catch Weight": 170,
    "Open Weight": 185,
}
MEAN_WC_LBS = 170  # fallback for unknown/missing


def _elo_expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def _elo_update(winner_elo: float, loser_elo: float, k_w: float, k_l: float):
    """Asymmetric update: winner and loser can carry different K-factors."""
    e_w = _elo_expected(winner_elo, loser_elo)
    return winner_elo + k_w * (1 - e_w), loser_elo - k_l * (1 - e_w)


def _k_factor(fights_before: int, is_ko: bool, is_sub: bool) -> float:
    """
    Improvement #3 — Performance-based K:
      KO/TKO=48, Sub=40, Decision=24  (finish quality matters)
    Improvement #4 — Provisional K for new fighters:
      < 5 fights  -> K *= 2.0   (rating calibrates fast)
      5-10 fights -> K *= 1.5
    """
    base_k = 48 if is_ko else (40 if is_sub else 24)
    if fights_before < 5:
        return min(80, base_k * 2.0)
    if fights_before < 10:
        return min(64, base_k * 1.5)
    return float(base_k)


# ==================================================================
# SECTION 4 -- CHRONOLOGICAL FEATURE ENGINEERING
# ==================================================================
#
#  CRITICAL: for each fight we capture all features BEFORE updating
#  ELO / fight history.  This prevents any form of data leakage.

print(f"\n[FEAT] Building features chronologically ...")

# -- Live state dicts --------------------------------------------
elo_global  = defaultdict(lambda: ELO_BASE)   # fighter -> overall ELO
elo_wc      = defaultdict(lambda: ELO_BASE)   # (fighter, wc) -> division ELO
fight_hist  = defaultdict(list)               # fighter -> list of past fight dicts
elo_log     = defaultdict(list)               # fighter -> [(date, elo)] for plotting


def _safe(val):
    """Return 0.0 for None / NaN, else float."""
    try:
        v = float(val)
        return 0.0 if np.isnan(v) else v
    except Exception:
        return 0.0


def _method_flags(method):
    m = str(method).upper() if not pd.isna(method) else ""
    is_ko  = "KO" in m or "TKO" in m
    is_sub = "SUB" in m
    is_dec = "DEC" in m
    return is_ko, is_sub, is_dec


def _rolling(hist, n=None):
    """Rolling stats from the last n fights (all fights if n is None).
    Improvement #2: also computes avg_opp_elo (opponent quality / SOS).
    """
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


def _streak(hist, won_val: bool) -> int:
    streak = 0
    for x in reversed(hist):
        if bool(x["w"]) == won_val:
            streak += 1
        else:
            break
    return streak


def _h2h(me: str, opp: str):
    h = fight_hist[me]
    wins   = sum(1 for x in h if x["opp"] == opp and     x["w"])
    losses = sum(1 for x in h if x["opp"] == opp and not x["w"])
    return wins, losses


def _age_at(dob, date):
    if pd.isna(dob) or pd.isna(date):
        return np.nan
    return (date - dob).days / 365.25


# -- Randomise fighter1 / fighter2 presentation -------------------
#  Inspection shows fighter1 is almost always the winner in the raw
#  data (scraper put the winner first).  Without randomisation the
#  model trivially learns "always predict fighter1" and reports ~100%
#  train accuracy but has no real predictive power.
#  We flip the presentation for a reproducible 50% of fights.
rng        = np.random.RandomState(42)
swap_flags = rng.rand(len(fights_raw)) < 0.5

feature_rows = []

for i, row in fights_raw.iterrows():
    orig_f1     = row["fighter1_name"]
    orig_f2     = row["fighter2_name"]
    winner_name = row["winner"]
    date        = row["event_date"]
    wc          = str(row.get("weight_class", "Unknown"))
    method      = row.get("method", "")

    # Apply swap
    if swap_flags[i]:
        f1, f2 = orig_f2, orig_f1
    else:
        f1, f2 = orig_f1, orig_f2

    # -- PRE-FIGHT ELO -------------------------------------------
    f1_elo    = elo_global[f1]
    f2_elo    = elo_global[f2]
    f1_elo_wc = elo_wc[(f1, wc)]
    f2_elo_wc = elo_wc[(f2, wc)]

    # -- HISTORICAL STATS ----------------------------------------
    h1 = fight_hist[f1]
    h2 = fight_hist[f2]

    s1_10  = _rolling(h1, 10)
    s1_25  = _rolling(h1, 25)
    s1_all = _rolling(h1)
    s2_10  = _rolling(h2, 10)
    s2_25  = _rolling(h2, 25)
    s2_all = _rolling(h2)

    # Improvement #4: track fight counts for provisional K (computed before update)
    f1_fights_before = len(h1)
    f2_fights_before = len(h2)

    f1_w_streak = _streak(h1, True)
    f1_l_streak = _streak(h1, False)
    f2_w_streak = _streak(h2, True)
    f2_l_streak = _streak(h2, False)

    h2h_f1w, h2h_f1l = _h2h(f1, f2)

    # Days since last fight (ring-rust proxy)
    f1_last = h1[-1]["date"] if h1 else None
    f2_last = h2[-1]["date"] if h2 else None
    f1_rest = _safe((date - f1_last).days) if f1_last else 365.0
    f2_rest = _safe((date - f2_last).days) if f2_last else 365.0

    # -- PHYSICAL ATTRIBUTES -------------------------------------
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

    f1_age = _age_at(f1_dob, date)
    f2_age = _age_at(f2_dob, date)

    def _diff(a, b):
        if np.isnan(a) or np.isnan(b):
            return np.nan
        return a - b

    # -- ASSEMBLE FEATURE DICT -----------------------------------
    feat = {}

    # ELO (article's #1 predictor — we expect same here)
    feat["elo_diff"]    = f1_elo    - f2_elo
    feat["elo_wc_diff"] = f1_elo_wc - f2_elo_wc
    feat["f1_elo"]      = f1_elo
    feat["f2_elo"]      = f2_elo
    feat["sum_elo"]     = f1_elo + f2_elo   # proxy for overall fight quality / era

    # Improvement #1: Weight class (ordinal by weight in lbs)
    feat["wc_lbs"] = float(WC_WEIGHT.get(wc, MEAN_WC_LBS))

    # Improvement #2: Opponent quality / Strength of Schedule
    feat["avg_opp_elo_diff_10"] = s1_10["avg_opp_elo"] - s2_10["avg_opp_elo"]
    feat["avg_opp_elo_diff_25"] = s1_25["avg_opp_elo"] - s2_25["avg_opp_elo"]
    feat["f1_avg_opp_elo_25"]   = s1_25["avg_opp_elo"]
    feat["f2_avg_opp_elo_25"]   = s2_25["avg_opp_elo"]

    # Physical
    feat["height_diff"] = _diff(f1_h,   f2_h)
    feat["reach_diff"]  = _diff(f1_r,   f2_r)
    feat["age_diff"]    = _diff(f1_age, f2_age)
    feat["f1_age"]      = f1_age
    feat["f2_age"]      = f2_age

    # Stance (one-hot encoded)
    feat["f1_orthodox"]     = int(f1_stance == "Orthodox")
    feat["f2_orthodox"]     = int(f2_stance == "Orthodox")
    feat["f1_southpaw"]     = int(f1_stance == "Southpaw")
    feat["f2_southpaw"]     = int(f2_stance == "Southpaw")
    feat["ortho_vs_south"]  = int(
        (f1_stance == "Orthodox" and f2_stance == "Southpaw") or
        (f1_stance == "Southpaw" and f2_stance == "Orthodox")
    )

    # Win rates at three windows
    feat["wr_diff_10"]  = s1_10["win_rate"]  - s2_10["win_rate"]
    feat["wr_diff_25"]  = s1_25["win_rate"]  - s2_25["win_rate"]
    feat["wr_diff_all"] = s1_all["win_rate"] - s2_all["win_rate"]

    # Finish-rate differentials
    feat["ko_rate_diff"]  = s1_25["ko_rate"]  - s2_25["ko_rate"]
    feat["sub_rate_diff"] = s1_25["sub_rate"] - s2_25["sub_rate"]
    feat["dec_rate_diff"] = s1_25["dec_rate"] - s2_25["dec_rate"]

    # Historical in-fight performance averages
    feat["sig_diff"]  = s1_25["avg_sig"]  - s2_25["avg_sig"]
    feat["td_diff"]   = s1_25["avg_td"]   - s2_25["avg_td"]
    feat["ctrl_diff"] = s1_25["avg_ctrl"] - s2_25["avg_ctrl"]

    # Experience
    feat["exp_diff"]  = s1_all["count"] - s2_all["count"]
    feat["f1_fights"] = s1_all["count"]
    feat["f2_fights"] = s2_all["count"]

    # Streaks
    feat["win_streak_diff"]  = f1_w_streak - f2_w_streak
    feat["lose_streak_diff"] = f1_l_streak - f2_l_streak
    feat["f1_win_streak"]    = f1_w_streak
    feat["f2_win_streak"]    = f2_w_streak

    # Head-to-head
    feat["h2h_diff"] = h2h_f1w - h2h_f1l
    feat["h2h_met"]  = int((h2h_f1w + h2h_f1l) > 0)

    # Rest / activity
    feat["rest_diff"]     = f1_rest - f2_rest
    feat["f1_rest_days"]  = f1_rest
    feat["f2_rest_days"]  = f2_rest

    # Target & metadata (excluded from X matrix later)
    feat["target"]     = int(winner_name == f1)
    feat["event_date"] = date
    feat["f1_name"]    = f1
    feat["f2_name"]    = f2

    feature_rows.append(feat)

    # -- UPDATE STATE AFTER RECORDING FEATURES -------------------
    f1_won = (winner_name == f1)
    ko_flag, sub_flag, dec_flag = _method_flags(method)

    # Map per-fighter stats from original CSV columns to actual fighters
    #  (CSV always labels orig_f1 / orig_f2 regardless of our swap)
    csv_stats = {
        orig_f1: (
            _safe(row.get("f1_sig_str_landed")),
            _safe(row.get("f1_td_landed")),
            _safe(row.get("f1_ctrl_sec")),
        ),
        orig_f2: (
            _safe(row.get("f2_sig_str_landed")),
            _safe(row.get("f2_td_landed")),
            _safe(row.get("f2_ctrl_sec")),
        ),
    }

    f1_sig,  f1_td,  f1_ctrl  = csv_stats.get(f1, (0.0, 0.0, 0.0))
    f2_sig,  f2_td,  f2_ctrl  = csv_stats.get(f2, (0.0, 0.0, 0.0))

    # Store opp_elo at fight time for opponent quality tracking (Improvement #2)
    fight_hist[f1].append(dict(
        date=date, opp=f2, opp_elo=f2_elo,
        w=f1_won, ko=ko_flag, sub=sub_flag, dec=dec_flag,
        sig=f1_sig, td=f1_td, ctrl=f1_ctrl,
    ))
    fight_hist[f2].append(dict(
        date=date, opp=f1, opp_elo=f1_elo,
        w=not f1_won, ko=ko_flag, sub=sub_flag, dec=dec_flag,
        sig=f2_sig, td=f2_td, ctrl=f2_ctrl,
    ))

    # Improvements #3+#4: performance-based + provisional K-factors
    if f1_won:
        k_w = _k_factor(f1_fights_before, ko_flag, sub_flag)
        k_l = _k_factor(f2_fights_before, ko_flag, sub_flag)
        new_w, new_l       = _elo_update(f1_elo,    f2_elo,    k_w, k_l)
        new_wc_w, new_wc_l = _elo_update(f1_elo_wc, f2_elo_wc, k_w, k_l)
        elo_global[f1], elo_global[f2]      = new_w,    new_l
        elo_wc[(f1, wc)], elo_wc[(f2, wc)] = new_wc_w, new_wc_l
    else:
        k_w = _k_factor(f2_fights_before, ko_flag, sub_flag)
        k_l = _k_factor(f1_fights_before, ko_flag, sub_flag)
        new_w, new_l       = _elo_update(f2_elo,    f1_elo,    k_w, k_l)
        new_wc_w, new_wc_l = _elo_update(f2_elo_wc, f1_elo_wc, k_w, k_l)
        elo_global[f2], elo_global[f1]      = new_w,    new_l
        elo_wc[(f2, wc)], elo_wc[(f1, wc)] = new_wc_w, new_wc_l

    # Log ELO trajectory for plotting
    elo_log[f1].append((date, elo_global[f1]))
    elo_log[f2].append((date, elo_global[f2]))

print(f"[FEAT] Done — {len(feature_rows):,} feature rows generated")


# ==================================================================
# SECTION 5 -- TRAIN / VALIDATE / TEST SPLIT  (CHRONOLOGICAL)
# ==================================================================

df = pd.DataFrame(feature_rows)

target_counts = df["target"].value_counts()
print(f"\n[SPLIT] Target balance: f1 wins={target_counts.get(1,0):,}  "
      f"f2 wins={target_counts.get(0,0):,}  "
      f"({target_counts.get(1,0)/len(df)*100:.1f}% / "
      f"{target_counts.get(0,0)/len(df)*100:.1f}%)")

TRAIN_END = pd.Timestamp("2022-12-31")
VAL_END   = pd.Timestamp("2023-12-31")

train_df = df[df["event_date"] <= TRAIN_END]
val_df   = df[(df["event_date"] > TRAIN_END) & (df["event_date"] <= VAL_END)]
test_df  = df[df["event_date"] > VAL_END]

for label, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    print(
        f"[SPLIT] {label:<6}: {len(split):>5,} fights  "
        f"({split['event_date'].min().date()} -> {split['event_date'].max().date()})"
    )

META_COLS    = {"target", "event_date", "f1_name", "f2_name"}
FEATURE_COLS = [c for c in df.columns if c not in META_COLS]

print(f"\n[FEAT] Using {len(FEATURE_COLS)} features")


def _xy(split):
    return split[FEATURE_COLS].copy(), split["target"].values


X_train, y_train = _xy(train_df)
X_val,   y_val   = _xy(val_df)
X_test,  y_test  = _xy(test_df)

# Impute medians (fitted on train only)
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)
X_val_imp   = imputer.transform(X_val)
X_test_imp  = imputer.transform(X_test)

# Scale (fitted on train only — NN needs this, tree models ignore it)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_imp)
X_val_sc   = scaler.transform(X_val_imp)
X_test_sc  = scaler.transform(X_test_imp)


# ==================================================================
# SECTION 6 -- MODEL TRAINING & EVALUATION
# ==================================================================

def _eval(name, model, Xv, yv, Xt, yt, pv=None, pt=None):
    """Print metrics and return result dict."""
    pred_v = model.predict(Xv)
    pred_t = model.predict(Xt)

    acc_v  = accuracy_score(yv, pred_v)
    acc_t  = accuracy_score(yt, pred_t)
    auc_v  = roc_auc_score(yv, pv) if pv is not None else float("nan")
    auc_t  = roc_auc_score(yt, pt) if pt is not None else float("nan")

    print(f"\n{'-'*50}")
    print(f"  {name}")
    print(f"{'-'*50}")
    print(f"  Validate — Accuracy: {acc_v:.4f}  |  AUC: {auc_v:.4f}")
    print(f"  Test     — Accuracy: {acc_t:.4f}  |  AUC: {auc_t:.4f}")
    print(f"\n  Test Classification Report:")
    print(classification_report(yt, pred_t, target_names=["f2 wins", "f1 wins"]))

    return dict(
        name=name,
        val_acc=acc_v,  test_acc=acc_t,
        val_auc=auc_v,  test_auc=auc_t,
        test_pred=pred_t, test_proba=pt,
    )


results = []

# -- 1. Decision Tree --------------------------------------------
print(f"\n{SEP}\n  1 / 4  DECISION TREE\n{SEP}")
dt = DecisionTreeClassifier(
    max_depth=8, min_samples_leaf=20, random_state=42
)
dt.fit(X_train_imp, y_train)
results.append(_eval(
    "Decision Tree", dt,
    X_val_imp, y_val, X_test_imp, y_test,
    dt.predict_proba(X_val_imp)[:, 1],
    dt.predict_proba(X_test_imp)[:, 1],
))

# -- 2. Random Forest --------------------------------------------
print(f"\n{SEP}\n  2 / 4  RANDOM FOREST\n{SEP}")
rf = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_leaf=10,
    n_jobs=-1, random_state=42,
)
rf.fit(X_train_imp, y_train)
results.append(_eval(
    "Random Forest", rf,
    X_val_imp, y_val, X_test_imp, y_test,
    rf.predict_proba(X_val_imp)[:, 1],
    rf.predict_proba(X_test_imp)[:, 1],
))

# -- 3. XGBoost --------------------------------------------------
print(f"\n{SEP}\n  3 / 4  XGBOOST\n{SEP}")
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric="logloss",
    early_stopping_rounds=30,
    random_state=42,
    verbosity=0,
)
xgb_model.fit(
    X_train_imp, y_train,
    eval_set=[(X_val_imp, y_val)],
    verbose=False,
)
results.append(_eval(
    "XGBoost", xgb_model,
    X_val_imp, y_val, X_test_imp, y_test,
    xgb_model.predict_proba(X_val_imp)[:, 1],
    xgb_model.predict_proba(X_test_imp)[:, 1],
))

# -- 4. Neural Network (MLP) -------------------------------------
print(f"\n{SEP}\n  4 / 4  NEURAL NETWORK (MLP)\n{SEP}")
nn = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation="relu",
    solver="adam",
    alpha=0.01,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
)
nn.fit(X_train_sc, y_train)
results.append(_eval(
    "Neural Network", nn,
    X_val_sc, y_val, X_test_sc, y_test,
    nn.predict_proba(X_val_sc)[:, 1],
    nn.predict_proba(X_test_sc)[:, 1],
))


# -- 5. XGBoost Hyperparameter Tuning (Random Search) ------------
#  Improvement #5: 60-trial random search on the validation set.
#  All evaluation is strictly on the held-out test set at the end —
#  the val set is ONLY used for early stopping and param selection.
print(f"\n{SEP}\n  5 / 5  XGBOOST TUNED (Random Search, 60 trials)\n{SEP}")

import random as _rand
_rand.seed(42)

_param_space = {
    "max_depth":        [3, 4, 5, 6, 7],
    "learning_rate":    [0.01, 0.03, 0.05, 0.08, 0.10],
    "min_child_weight": [5, 10, 15, 20, 30],
    "subsample":        [0.60, 0.70, 0.80, 0.90],
    "colsample_bytree": [0.60, 0.70, 0.80, 0.90],
    "reg_alpha":        [0.0, 0.1, 0.5, 1.0],
    "reg_lambda":       [0.5, 1.0, 2.0, 5.0],
}

_best_val_auc  = -1.0
_best_xgb      = None
_best_params   = None
N_TRIALS       = 60

print(f"[TUNE] Searching {N_TRIALS} random combinations ...")
for trial in range(N_TRIALS):
    params = {k: _rand.choice(v) for k, v in _param_space.items()}
    _cand = xgb.XGBClassifier(
        n_estimators=1000,
        eval_metric="logloss",
        early_stopping_rounds=25,
        random_state=42,
        verbosity=0,
        **params,
    )
    _cand.fit(
        X_train_imp, y_train,
        eval_set=[(X_val_imp, y_val)],
        verbose=False,
    )
    _val_auc = roc_auc_score(y_val, _cand.predict_proba(X_val_imp)[:, 1])
    if _val_auc > _best_val_auc:
        _best_val_auc = _val_auc
        _best_xgb     = _cand
        _best_params  = params
    if (trial + 1) % 15 == 0:
        print(f"  trial {trial+1:>3}/{N_TRIALS}  best val AUC: {_best_val_auc:.4f}")

print(f"\n[TUNE] Best val AUC: {_best_val_auc:.4f}")
print(f"[TUNE] Best params:  {_best_params}")
results.append(_eval(
    "XGBoost (Tuned)", _best_xgb,
    X_val_imp, y_val, X_test_imp, y_test,
    _best_xgb.predict_proba(X_val_imp)[:, 1],
    _best_xgb.predict_proba(X_test_imp)[:, 1],
))


# ==================================================================
# SECTION 7 -- SUMMARY TABLE
# ==================================================================

print(f"\n{SEP}")
print("  RESULTS SUMMARY")
print(SEP)

summary = pd.DataFrame([{
    "Model":    r["name"],
    "Val Acc":  f"{r['val_acc']:.4f}",
    "Test Acc": f"{r['test_acc']:.4f}",
    "Val AUC":  f"{r['val_auc']:.4f}",
    "Test AUC": f"{r['test_auc']:.4f}",
} for r in results])
print(summary.to_string(index=False))


# ==================================================================
# SECTION 8 -- VISUALISATIONS
# ==================================================================

# -- 8a. Model comparison (accuracy + AUC) -----------------------
names     = [r["name"] for r in results]
val_accs  = [r["val_acc"]  for r in results]
test_accs = [r["test_acc"] for r in results]
val_aucs  = [r["val_auc"]  for r in results]
test_aucs = [r["test_auc"] for r in results]
x = np.arange(len(names))
w = 0.35

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

for ax, metric, vals_v, vals_t, title in [
    (axes[0], "Accuracy",  val_accs,  test_accs, "Accuracy"),
    (axes[1], "ROC-AUC",   val_aucs,  test_aucs, "ROC-AUC"),
]:
    ax.bar(x - w / 2, vals_v, w, label="Validation", color="#4C72B0")
    ax.bar(x + w / 2, vals_t, w, label="Test",       color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel(metric)
    ax.set_title(f"Model {title} — Val vs Test")
    ax.legend()
    ax.axhline(0.5, ls="--", color="grey", alpha=0.5, label="Random baseline")

plt.suptitle("UFC Fight Prediction — Model Comparison", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/model_comparison.png", dpi=150)
plt.close()
print("\n[PLOT] Saved plots/model_comparison.png")


# -- 8b. Feature importance — XGBoost Tuned ---------------------
feat_imp = (
    pd.DataFrame({"feature": FEATURE_COLS,
                  "importance": _best_xgb.feature_importances_})
    .sort_values("importance", ascending=False)
    .head(20)
)

plt.figure(figsize=(10, 7))
sns.barplot(data=feat_imp, x="importance", y="feature", palette="viridis")
plt.title("XGBoost (Tuned) — Top 20 Feature Importances", fontsize=12)
plt.tight_layout()
plt.savefig("plots/feature_importance.png", dpi=150)
plt.close()
print("[PLOT] Saved plots/feature_importance.png")


# -- 8c. Confusion matrices (all 5 models) -----------------------
fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
for ax, r in zip(axes, results):
    cm = confusion_matrix(y_test, r["test_pred"])
    sns.heatmap(
        cm, annot=True, fmt="d", ax=ax, cmap="Blues",
        xticklabels=["f2 wins", "f1 wins"],
        yticklabels=["f2 wins", "f1 wins"],
    )
    ax.set_title(f"{r['name']}\nAcc={r['test_acc']:.3f}", fontsize=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.suptitle("Confusion Matrices — Test Set", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/confusion_matrices.png", dpi=150)
plt.close()
print("[PLOT] Saved plots/confusion_matrices.png")


# -- 8d. ELO leaderboard — top 20 fighters -----------------------
print(f"\n{SEP}")
print("  CURRENT ELO LEADERBOARD — TOP 20")
print(SEP)

elo_series = (
    pd.Series(dict(elo_global))
    .sort_values(ascending=False)
    .head(20)
)

for rank, (name, rating) in enumerate(elo_series.items(), 1):
    total_fights = len(fight_hist.get(name, []))
    print(f"  {rank:>2}. {name:<30}  ELO: {rating:>7.1f}  ({total_fights} fights)")

# Bar chart
fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#C0392B" if i == 0 else "#4C72B0" for i in range(len(elo_series))]
ax.barh(elo_series.index[::-1], elo_series.values[::-1], color=colors[::-1])
ax.axvline(ELO_BASE, ls="--", color="grey", alpha=0.7, label=f"Starting ELO ({ELO_BASE})")
ax.set_xlabel("ELO Rating")
ax.set_title("UFC ELO Leaderboard — Top 20 Fighters (All-Time)", fontsize=12, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("plots/elo_leaderboard.png", dpi=150)
plt.close()
print("\n[PLOT] Saved plots/elo_leaderboard.png")


# -- 8e. ELO trajectories — top 5 fighters -----------------------
top5 = elo_series.head(5).index.tolist()

plt.figure(figsize=(13, 6))
palette = plt.cm.tab10.colors

for idx, fighter in enumerate(top5):
    log = elo_log.get(fighter, [])
    if not log:
        continue
    dates  = [x[0] for x in log]
    ratings = [x[1] for x in log]
    plt.plot(dates, ratings, label=fighter, linewidth=2, color=palette[idx])

plt.axhline(ELO_BASE, ls="--", color="grey", alpha=0.5, label=f"Starting ELO ({ELO_BASE})")
plt.xlabel("Date")
plt.ylabel("Overall ELO Rating")
plt.title(
    "ELO Rating Trajectories — Top 5 Fighters\n"
    "(each point = one fight; upward = winning; downward = losing)",
    fontsize=11, fontweight="bold",
)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("plots/elo_trajectories.png", dpi=150)
plt.close()
print("[PLOT] Saved plots/elo_trajectories.png")


# ==================================================================
# SECTION 9 -- SAVE ARTIFACTS FOR predict.py
# ==================================================================
import joblib

print(f"\n{SEP}")
print("  SAVING MODEL ARTIFACTS")
print(SEP)

joblib.dump(_best_xgb, "xgb_model.pkl")
joblib.dump(imputer,   "imputer.pkl")

artifacts = {
    "elo_global":   dict(elo_global),
    "elo_wc":       {str(k): v for k, v in elo_wc.items()},  # stringify tuple keys
    "fight_hist":   dict(fight_hist),
    "fighter_phys": fighter_phys,
    "FEATURE_COLS": FEATURE_COLS,
    "WC_WEIGHT":    WC_WEIGHT,
    "MEAN_WC_LBS":  MEAN_WC_LBS,
    "ELO_BASE":     ELO_BASE,
}
joblib.dump(artifacts, "model_artifacts.pkl")

print("[SAVE] xgb_model.pkl       — tuned XGBoost classifier")
print("[SAVE] imputer.pkl         — median imputer (fitted on train)")
print("[SAVE] model_artifacts.pkl — ELO ratings, fight history, fighter physical data")
print("\n       Run  python predict.py  to make fight predictions.")

# ==================================================================
# DONE
# ==================================================================
print(f"\n{SEP}")
print("  ALL DONE  — check the plots/ folder for visualisations")
print(SEP)
