#!/usr/bin/env python3
"""
UFC Fight Outcome Prediction Model (v2)

New in v2:
- Full dataset mirroring option (every fight represented from both fighter perspectives)
- Probability-first evaluation (log loss, Brier, ECE)
- Isotonic regression calibration on validation probabilities
- Calibrated artifact saving for downstream CLI inference
"""

import os
import argparse
import random
from collections import defaultdict

import numpy as np
import pandas as pd

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
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    log_loss,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

import xgboost as xgb
import joblib


np.random.seed(42)
random.seed(42)

SEP = "=" * 70
PLOTS_DIR = "plots_v2"
os.makedirs(PLOTS_DIR, exist_ok=True)

ELO_BASE = 1500
WC_WEIGHT = {
    "Strawweight": 115,
    "Women's Strawweight": 115,
    "Flyweight": 125,
    "Women's Flyweight": 125,
    "Bantamweight": 135,
    "Women's Bantamweight": 135,
    "Featherweight": 145,
    "Women's Featherweight": 145,
    "Lightweight": 155,
    "Welterweight": 170,
    "Middleweight": 185,
    "Light Heavyweight": 205,
    "Heavyweight": 265,
    "Super Heavyweight": 265,
    "Catch Weight": 170,
    "Open Weight": 185,
}
MEAN_WC_LBS = 170


def _height_in(h):
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
        feet = int(parts[0].strip())
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


def _elo_expected(ra, rb):
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def _elo_update(winner_elo, loser_elo, k_w, k_l):
    e_w = _elo_expected(winner_elo, loser_elo)
    return winner_elo + k_w * (1 - e_w), loser_elo - k_l * (1 - e_w)


def _k_factor(fights_before, is_ko, is_sub):
    base_k = 48 if is_ko else (40 if is_sub else 24)
    if fights_before < 5:
        return min(80, base_k * 2.0)
    if fights_before < 10:
        return min(64, base_k * 1.5)
    return float(base_k)


def _method_flags(method):
    m = str(method).upper() if not pd.isna(method) else ""
    is_ko = "KO" in m or "TKO" in m
    is_sub = "SUB" in m
    is_dec = "DEC" in m
    return is_ko, is_sub, is_dec


def _rolling(hist, n=None):
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
            "avg_opp_elo": float(ELO_BASE),
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
        "avg_opp_elo": float(np.mean([x.get("opp_elo", ELO_BASE) for x in h])),
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
    h = fight_hist[me]
    wins = sum(1 for x in h if x["opp"] == opp and x["w"])
    losses = sum(1 for x in h if x["opp"] == opp and not x["w"])
    return wins, losses


def _age_at(dob, date):
    if pd.isna(dob) or pd.isna(date):
        return np.nan
    return (date - dob).days / 365.25


def _ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bucket_ids = np.digitize(y_prob, bins, right=True)
    ece = 0.0
    n = len(y_true)
    for b in range(1, n_bins + 1):
        idx = bucket_ids == b
        if not np.any(idx):
            continue
        conf = float(np.mean(y_prob[idx]))
        acc = float(np.mean(y_true[idx]))
        weight = float(np.sum(idx)) / n
        ece += weight * abs(acc - conf)
    return ece


def build_features_dataframe(fights_raw, fighter_phys, build_mode):
    elo_global = defaultdict(lambda: ELO_BASE)
    elo_wc = defaultdict(lambda: ELO_BASE)
    fight_hist = defaultdict(list)
    elo_log = defaultdict(list)

    rng = np.random.RandomState(42)
    swap_flags = rng.rand(len(fights_raw)) < 0.5
    feature_rows = []

    for i, row in fights_raw.iterrows():
        orig_f1 = row["fighter1_name"]
        orig_f2 = row["fighter2_name"]
        winner_name = row["winner"]
        date = row["event_date"]
        wc = str(row.get("weight_class", "Unknown"))
        method = row.get("method", "")

        if build_mode == "full_mirror":
            perspectives = [
                (orig_f1, orig_f2, int(winner_name == orig_f1)),
                (orig_f2, orig_f1, int(winner_name == orig_f2)),
            ]
        else:
            if swap_flags[i]:
                f1, f2 = orig_f2, orig_f1
            else:
                f1, f2 = orig_f1, orig_f2
            perspectives = [(f1, f2, int(winner_name == f1))]

        for f1, f2, target in perspectives:
            f1_elo = elo_global[f1]
            f2_elo = elo_global[f2]
            f1_elo_wc = elo_wc[(f1, wc)]
            f2_elo_wc = elo_wc[(f2, wc)]

            h1 = fight_hist[f1]
            h2 = fight_hist[f2]

            s1_10 = _rolling(h1, 10)
            s1_25 = _rolling(h1, 25)
            s1_all = _rolling(h1)
            s2_10 = _rolling(h2, 10)
            s2_25 = _rolling(h2, 25)
            s2_all = _rolling(h2)

            f1_w_streak = _streak(h1, True)
            f1_l_streak = _streak(h1, False)
            f2_w_streak = _streak(h2, True)
            f2_l_streak = _streak(h2, False)
            h2h_f1w, h2h_f1l = _h2h(fight_hist, f1, f2)

            f1_last = h1[-1]["date"] if h1 else None
            f2_last = h2[-1]["date"] if h2 else None
            f1_rest = _safe((date - f1_last).days) if f1_last else 365.0
            f2_rest = _safe((date - f2_last).days) if f2_last else 365.0

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

            f1_age = _age_at(f1_dob, date)
            f2_age = _age_at(f2_dob, date)

            feat = {}
            feat["elo_diff"] = f1_elo - f2_elo
            feat["elo_wc_diff"] = f1_elo_wc - f2_elo_wc
            feat["f1_elo"] = f1_elo
            feat["f2_elo"] = f2_elo
            feat["sum_elo"] = f1_elo + f2_elo
            feat["wc_lbs"] = float(WC_WEIGHT.get(wc, MEAN_WC_LBS))

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

            feat["target"] = target
            feat["event_date"] = date
            feat["f1_name"] = f1
            feat["f2_name"] = f2
            feature_rows.append(feat)

        # update state exactly once per real fight
        f1_elo = elo_global[orig_f1]
        f2_elo = elo_global[orig_f2]
        f1_elo_wc = elo_wc[(orig_f1, wc)]
        f2_elo_wc = elo_wc[(orig_f2, wc)]
        h1 = fight_hist[orig_f1]
        h2 = fight_hist[orig_f2]
        f1_fights_before = len(h1)
        f2_fights_before = len(h2)

        ko_flag, sub_flag, dec_flag = _method_flags(method)
        orig_f1_won = winner_name == orig_f1

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
        f1_sig, f1_td, f1_ctrl = csv_stats.get(orig_f1, (0.0, 0.0, 0.0))
        f2_sig, f2_td, f2_ctrl = csv_stats.get(orig_f2, (0.0, 0.0, 0.0))

        fight_hist[orig_f1].append({
            "date": date,
            "opp": orig_f2,
            "opp_elo": f2_elo,
            "w": orig_f1_won,
            "ko": ko_flag,
            "sub": sub_flag,
            "dec": dec_flag,
            "sig": f1_sig,
            "td": f1_td,
            "ctrl": f1_ctrl,
        })
        fight_hist[orig_f2].append({
            "date": date,
            "opp": orig_f1,
            "opp_elo": f1_elo,
            "w": not orig_f1_won,
            "ko": ko_flag,
            "sub": sub_flag,
            "dec": dec_flag,
            "sig": f2_sig,
            "td": f2_td,
            "ctrl": f2_ctrl,
        })

        if orig_f1_won:
            k_w = _k_factor(f1_fights_before, ko_flag, sub_flag)
            k_l = _k_factor(f2_fights_before, ko_flag, sub_flag)
            new_w, new_l = _elo_update(f1_elo, f2_elo, k_w, k_l)
            new_wc_w, new_wc_l = _elo_update(f1_elo_wc, f2_elo_wc, k_w, k_l)
            elo_global[orig_f1], elo_global[orig_f2] = new_w, new_l
            elo_wc[(orig_f1, wc)], elo_wc[(orig_f2, wc)] = new_wc_w, new_wc_l
        else:
            k_w = _k_factor(f2_fights_before, ko_flag, sub_flag)
            k_l = _k_factor(f1_fights_before, ko_flag, sub_flag)
            new_w, new_l = _elo_update(f2_elo, f1_elo, k_w, k_l)
            new_wc_w, new_wc_l = _elo_update(f2_elo_wc, f1_elo_wc, k_w, k_l)
            elo_global[orig_f2], elo_global[orig_f1] = new_w, new_l
            elo_wc[(orig_f2, wc)], elo_wc[(orig_f1, wc)] = new_wc_w, new_wc_l

        elo_log[orig_f1].append((date, elo_global[orig_f1]))
        elo_log[orig_f2].append((date, elo_global[orig_f2]))

    df = pd.DataFrame(feature_rows)
    return df, elo_global, elo_wc, fight_hist, elo_log


def evaluate_model(name, model, Xv, yv, Xt, yt, pv, pt):
    pred_v = (pv >= 0.5).astype(int)
    pred_t = (pt >= 0.5).astype(int)

    out = {
        "name": name,
        "val_acc": accuracy_score(yv, pred_v),
        "test_acc": accuracy_score(yt, pred_t),
        "val_auc": roc_auc_score(yv, pv),
        "test_auc": roc_auc_score(yt, pt),
        "val_logloss": log_loss(yv, pv),
        "test_logloss": log_loss(yt, pt),
        "val_brier": brier_score_loss(yv, pv),
        "test_brier": brier_score_loss(yt, pt),
        "val_ece": _ece(yv, pv),
        "test_ece": _ece(yt, pt),
        "test_pred": pred_t,
        "test_proba": pt,
    }

    print("\n" + "-" * 55)
    print(f"  {name}")
    print("-" * 55)
    print(f"  Val  Acc/AUC      : {out['val_acc']:.4f} / {out['val_auc']:.4f}")
    print(f"  Test Acc/AUC      : {out['test_acc']:.4f} / {out['test_auc']:.4f}")
    print(f"  Val  LogLoss/Brier: {out['val_logloss']:.4f} / {out['val_brier']:.4f}")
    print(f"  Test LogLoss/Brier: {out['test_logloss']:.4f} / {out['test_brier']:.4f}")
    print(f"  Val/Test ECE      : {out['val_ece']:.4f} / {out['test_ece']:.4f}")
    print("\n  Test Classification Report:")
    print(classification_report(yt, pred_t, target_names=["f2 wins", "f1 wins"]))
    return out


def main():
    parser = argparse.ArgumentParser(description="UFC model trainer (v2)")
    parser.add_argument(
        "--build-mode",
        choices=["full_mirror", "random_swap"],
        default="full_mirror",
        help="Data orientation strategy. full_mirror duplicates each fight in both directions.",
    )
    args = parser.parse_args()

    print(SEP)
    print(" UFC FIGHT OUTCOME PREDICTION MODEL (v2)")
    print(SEP)
    print(f"[CONFIG] build_mode={args.build_mode}")

    fights_raw = pd.read_csv("data/fights.csv")
    fighters_raw = pd.read_csv("data/fighters.csv")

    fights_raw["event_date"] = pd.to_datetime(fights_raw["event_date"], errors="coerce")
    fighters_raw["dob"] = pd.to_datetime(fighters_raw["dob"], errors="coerce")

    fights_raw = fights_raw.dropna(subset=["winner", "fighter1_name", "fighter2_name", "event_date"])
    fights_raw = fights_raw.sort_values("event_date").reset_index(drop=True)

    fighters_raw["height_in"] = fighters_raw["height"].apply(_height_in)
    fighters_raw["reach_in"] = fighters_raw["reach"].apply(_reach_in)
    fighters_raw["weight_lbs"] = fighters_raw["weight"].apply(_weight_lbs)

    fighter_phys = (
        fighters_raw
        .drop_duplicates(subset="fighter_name", keep="last")
        .set_index("fighter_name")[["height_in", "reach_in", "weight_lbs", "dob", "stance"]]
        .to_dict("index")
    )

    print(f"[DATA] fights={len(fights_raw):,}  fighters={len(fighters_raw):,}")
    print(f"[DATA] date_range={fights_raw['event_date'].min().date()} -> {fights_raw['event_date'].max().date()}")

    df, elo_global, elo_wc, fight_hist, elo_log = build_features_dataframe(
        fights_raw=fights_raw,
        fighter_phys=fighter_phys,
        build_mode=args.build_mode,
    )

    print(f"[FEAT] rows={len(df):,}")
    print(f"[FEAT] target_balance={df['target'].mean():.3f} (f1 win rate)")

    TRAIN_END = pd.Timestamp("2022-12-31")
    VAL_END = pd.Timestamp("2023-12-31")

    train_df = df[df["event_date"] <= TRAIN_END]
    val_df = df[(df["event_date"] > TRAIN_END) & (df["event_date"] <= VAL_END)]
    test_df = df[df["event_date"] > VAL_END]

    for label, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(
            f"[SPLIT] {label:<5} {len(split):>6,}  "
            f"({split['event_date'].min().date()} -> {split['event_date'].max().date()})"
        )

    META_COLS = {"target", "event_date", "f1_name", "f2_name"}
    FEATURE_COLS = [c for c in df.columns if c not in META_COLS]

    X_train = train_df[FEATURE_COLS].copy()
    y_train = train_df["target"].values
    X_val = val_df[FEATURE_COLS].copy()
    y_val = val_df["target"].values
    X_test = test_df[FEATURE_COLS].copy()
    y_test = test_df["target"].values

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_imp)
    X_val_sc = scaler.transform(X_val_imp)
    X_test_sc = scaler.transform(X_test_imp)

    results = []

    print("\n" + SEP)
    print(" 1 / 5 DECISION TREE")
    print(SEP)
    dt = DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, random_state=42)
    dt.fit(X_train_imp, y_train)
    pv = dt.predict_proba(X_val_imp)[:, 1]
    pt = dt.predict_proba(X_test_imp)[:, 1]
    results.append(evaluate_model("Decision Tree", dt, X_val_imp, y_val, X_test_imp, y_test, pv, pt))

    print("\n" + SEP)
    print(" 2 / 5 RANDOM FOREST")
    print(SEP)
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train_imp, y_train)
    pv = rf.predict_proba(X_val_imp)[:, 1]
    pt = rf.predict_proba(X_test_imp)[:, 1]
    results.append(evaluate_model("Random Forest", rf, X_val_imp, y_val, X_test_imp, y_test, pv, pt))

    print("\n" + SEP)
    print(" 3 / 5 XGBOOST")
    print(SEP)
    xgb_base = xgb.XGBClassifier(
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
    xgb_base.fit(X_train_imp, y_train, eval_set=[(X_val_imp, y_val)], verbose=False)
    pv = xgb_base.predict_proba(X_val_imp)[:, 1]
    pt = xgb_base.predict_proba(X_test_imp)[:, 1]
    results.append(evaluate_model("XGBoost", xgb_base, X_val_imp, y_val, X_test_imp, y_test, pv, pt))

    print("\n" + SEP)
    print(" 4 / 5 NEURAL NETWORK")
    print(SEP)
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
    pv = nn.predict_proba(X_val_sc)[:, 1]
    pt = nn.predict_proba(X_test_sc)[:, 1]
    results.append(evaluate_model("Neural Network", nn, X_val_sc, y_val, X_test_sc, y_test, pv, pt))

    print("\n" + SEP)
    print(" 5 / 5 XGBOOST TUNED (random search)")
    print(SEP)
    param_space = {
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.10],
        "min_child_weight": [5, 10, 15, 20, 30],
        "subsample": [0.60, 0.70, 0.80, 0.90],
        "colsample_bytree": [0.60, 0.70, 0.80, 0.90],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }

    best_val_auc = -1.0
    best_xgb = None
    best_params = None
    n_trials = 60

    for trial in range(n_trials):
        params = {k: random.choice(v) for k, v in param_space.items()}
        cand = xgb.XGBClassifier(
            n_estimators=1000,
            eval_metric="logloss",
            early_stopping_rounds=25,
            random_state=42,
            verbosity=0,
            **params,
        )
        cand.fit(X_train_imp, y_train, eval_set=[(X_val_imp, y_val)], verbose=False)
        val_auc = roc_auc_score(y_val, cand.predict_proba(X_val_imp)[:, 1])
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_xgb = cand
            best_params = params
        if (trial + 1) % 15 == 0:
            print(f"  trial {trial+1:>3}/{n_trials} best val AUC={best_val_auc:.4f}")

    print(f"[TUNE] best_params={best_params}")
    pv_raw = best_xgb.predict_proba(X_val_imp)[:, 1]
    pt_raw = best_xgb.predict_proba(X_test_imp)[:, 1]
    results.append(evaluate_model("XGBoost (Tuned)", best_xgb, X_val_imp, y_val, X_test_imp, y_test, pv_raw, pt_raw))

    print("\n" + SEP)
    print(" CALIBRATION: ISOTONIC ON VAL SET")
    print(SEP)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(pv_raw, y_val)
    pv_cal = calibrator.transform(pv_raw)
    pt_cal = calibrator.transform(pt_raw)

    results.append(
        evaluate_model(
            "XGBoost (Tuned + Isotonic)",
            best_xgb,
            X_val_imp,
            y_val,
            X_test_imp,
            y_test,
            pv_cal,
            pt_cal,
        )
    )

    print("\n" + SEP)
    print(" RESULTS SUMMARY")
    print(SEP)
    summary = pd.DataFrame([
        {
            "Model": r["name"],
            "ValAcc": round(r["val_acc"], 4),
            "TestAcc": round(r["test_acc"], 4),
            "ValAUC": round(r["val_auc"], 4),
            "TestAUC": round(r["test_auc"], 4),
            "ValLL": round(r["val_logloss"], 4),
            "TestLL": round(r["test_logloss"], 4),
            "ValBrier": round(r["val_brier"], 4),
            "TestBrier": round(r["test_brier"], 4),
            "ValECE": round(r["val_ece"], 4),
            "TestECE": round(r["test_ece"], 4),
        }
        for r in results
    ])
    print(summary.to_string(index=False))
    summary.to_csv("training_summary_v2.csv", index=False)

    # Model metric comparison
    names = [r["name"] for r in results]
    test_accs = [r["test_acc"] for r in results]
    test_aucs = [r["test_auc"] for r in results]
    test_ll = [r["test_logloss"] for r in results]
    test_brier = [r["test_brier"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    sns.barplot(x=names, y=test_accs, ax=axes[0, 0], color="#4C72B0")
    axes[0, 0].set_title("Test Accuracy")
    axes[0, 0].tick_params(axis="x", rotation=20)

    sns.barplot(x=names, y=test_aucs, ax=axes[0, 1], color="#55A868")
    axes[0, 1].set_title("Test AUC")
    axes[0, 1].tick_params(axis="x", rotation=20)

    sns.barplot(x=names, y=test_ll, ax=axes[1, 0], color="#C44E52")
    axes[1, 0].set_title("Test Log Loss (lower better)")
    axes[1, 0].tick_params(axis="x", rotation=20)

    sns.barplot(x=names, y=test_brier, ax=axes[1, 1], color="#8172B3")
    axes[1, 1].set_title("Test Brier Score (lower better)")
    axes[1, 1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "model_metrics_v2.png"), dpi=150)
    plt.close()

    # Reliability curve (raw vs isotonic on tuned model)
    frac_pos_raw, mean_pred_raw = calibration_curve(y_test, pt_raw, n_bins=10, strategy="quantile")
    frac_pos_cal, mean_pred_cal = calibration_curve(y_test, pt_cal, n_bins=10, strategy="quantile")

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.plot(mean_pred_raw, frac_pos_raw, marker="o", label="Tuned XGBoost (raw)")
    plt.plot(mean_pred_cal, frac_pos_cal, marker="o", label="Tuned XGBoost (isotonic)")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed win rate")
    plt.title("Reliability Curve (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "reliability_curve_v2.png"), dpi=150)
    plt.close()

    # Confusion matrices
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4))
    for ax, r in zip(np.atleast_1d(axes), results):
        cm = confusion_matrix(y_test, r["test_pred"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["f2 wins", "f1 wins"],
            yticklabels=["f2 wins", "f1 wins"],
        )
        ax.set_title(r["name"], fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrices_v2.png"), dpi=150)
    plt.close()

    print("\n" + SEP)
    print(" SAVING V2 ARTIFACTS")
    print(SEP)

    joblib.dump(best_xgb, "xgb_model_v2.pkl")
    joblib.dump(imputer, "imputer_v2.pkl")
    joblib.dump(calibrator, "isotonic_calibrator_v2.pkl")

    artifacts = {
        "elo_global": dict(elo_global),
        "elo_wc": {str(k): v for k, v in elo_wc.items()},
        "fight_hist": dict(fight_hist),
        "fighter_phys": fighter_phys,
        "FEATURE_COLS": FEATURE_COLS,
        "WC_WEIGHT": WC_WEIGHT,
        "MEAN_WC_LBS": MEAN_WC_LBS,
        "ELO_BASE": ELO_BASE,
        "build_mode": args.build_mode,
    }
    joblib.dump(artifacts, "model_artifacts_v2.pkl")

    print("[SAVE] xgb_model_v2.pkl")
    print("[SAVE] imputer_v2.pkl")
    print("[SAVE] isotonic_calibrator_v2.pkl")
    print("[SAVE] model_artifacts_v2.pkl")
    print("[SAVE] training_summary_v2.csv")

    print("\n" + SEP)
    print(" DONE - run: python predict_v2.py")
    print(SEP)


if __name__ == "__main__":
    main()
