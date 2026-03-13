#!/usr/bin/env python3
"""
European club soccer model pipeline (mirrored rows + calibration).

Design choice for multi-team player issue:
- Model is club-centric (team-level), not player identity-centric.
- Features are competition-aware + recency-aware, so transfers are absorbed
  by recent form without mixing national-team identity.
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone

_LOCAL_DEPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "NFL", "pydeps")
_LOCAL_DEPS = os.path.abspath(_LOCAL_DEPS)
if os.path.isdir(_LOCAL_DEPS) and _LOCAL_DEPS not in sys.path:
    sys.path.insert(0, _LOCAL_DEPS)

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

np.random.seed(42)
ELO_BASE = 1500.0
HOME_ADV = 60.0


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
        return {
            "wr": 0.5,
            "gf": 1.4,
            "ga": 1.4,
            "margin": 0.0,
            "opp_elo": ELO_BASE,
            "count": 0,
        }
    return {
        "wr": float(np.mean([x["win"] for x in h])),
        "gf": float(np.mean([x["gf"] for x in h])),
        "ga": float(np.mean([x["ga"] for x in h])),
        "margin": float(np.mean([x["margin"] for x in h])),
        "opp_elo": float(np.mean([x["opp_elo"] for x in h])),
        "count": len(h),
    }


def ece_score(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_prob, bins, right=True)
    out = 0.0
    n = len(y_true)
    for b in range(1, n_bins + 1):
        m = idx == b
        if not np.any(m):
            continue
        out += (np.sum(m) / n) * abs(np.mean(y_true[m]) - np.mean(y_prob[m]))
    return float(out)


def elo_expect(home_elo, away_elo):
    return 1.0 / (1.0 + 10.0 ** ((away_elo - (home_elo + HOME_ADV)) / 400.0))


def elo_update(home_elo, away_elo, home_win, mov, is_cup=False):
    exp_home = elo_expect(home_elo, away_elo)
    k = 24.0 if is_cup else 20.0
    mov_mult = ((abs(mov) + 1.0) ** 0.8) / (7.5 + 0.006 * abs(home_elo - away_elo))
    delta = k * mov_mult * ((1.0 if home_win else 0.0) - exp_home)
    return home_elo + delta, away_elo - delta


def build_feature_table(matches: pd.DataFrame) -> pd.DataFrame:
    overall_elo = defaultdict(lambda: ELO_BASE)
    comp_elo = defaultdict(lambda: ELO_BASE)

    overall_hist = defaultdict(lambda: deque(maxlen=60))
    comp_hist = defaultdict(lambda: deque(maxlen=40))

    last_date = {}
    comp_season_record = defaultdict(lambda: {"w": 0, "l": 0})

    rows = []

    for _, g in matches.iterrows():
        home = g["home_team"]
        away = g["away_team"]
        comp = g["competition_code"]
        season = int(g["season_start"])
        gd = g["match_date"]

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

        h_rest = (gd - last_date[home]).days if home in last_date else 7
        a_rest = (gd - last_date[away]).days if away in last_date else 7

        h5o = summarize(overall_hist[home], 5)
        h10o = summarize(overall_hist[home], 10)
        a5o = summarize(overall_hist[away], 5)
        a10o = summarize(overall_hist[away], 10)

        h5c = summarize(comp_hist[(home, comp)], 5)
        h10c = summarize(comp_hist[(home, comp)], 10)
        a5c = summarize(comp_hist[(away, comp)], 5)
        a10c = summarize(comp_hist[(away, comp)], 10)

        hr = comp_season_record[(home, comp, season)]
        ar = comp_season_record[(away, comp, season)]
        hg_n = hr["w"] + hr["l"]
        ag_n = ar["w"] + ar["l"]
        h_wr = hr["w"] / hg_n if hg_n > 0 else 0.5
        a_wr = ar["w"] / ag_n if ag_n > 0 else 0.5

        imp_h = implied_prob_decimal(g["b365_home_odds"])
        imp_d = implied_prob_decimal(g["b365_draw_odds"])
        imp_a = implied_prob_decimal(g["b365_away_odds"])
        nv_h, nv_d, nv_a = no_vig(imp_h, imp_d, imp_a)

        common = {
            "season_start": season,
            "match_date": gd,
            "competition_code": comp,
            "competition_name": g["competition_name"],
            "is_uefa_comp": int(comp in ["CL", "EL", "EC"]),
            "month": int(gd.month),
            "weekday": int(gd.dayofweek),
            "team_a": home,
            "team_b": away,
            "team_a_is_home": 1,
            "elo_overall_diff": h_elo_o - a_elo_o,
            "elo_comp_diff": h_elo_c - a_elo_c,
            "rest_diff": h_rest - a_rest,
            "team_a_rest": h_rest,
            "team_b_rest": a_rest,
            "team_a_b2b": int(h_rest <= 3),
            "team_b_b2b": int(a_rest <= 3),
            "season_wr_comp_diff": h_wr - a_wr,
            "wr5_overall_diff": h5o["wr"] - a5o["wr"],
            "wr10_overall_diff": h10o["wr"] - a10o["wr"],
            "margin10_overall_diff": h10o["margin"] - a10o["margin"],
            "gf10_overall_diff": h10o["gf"] - a10o["gf"],
            "ga10_overall_diff": h10o["ga"] - a10o["ga"],
            "wr5_comp_diff": h5c["wr"] - a5c["wr"],
            "wr10_comp_diff": h10c["wr"] - a10c["wr"],
            "margin10_comp_diff": h10c["margin"] - a10c["margin"],
            "gf10_comp_diff": h10c["gf"] - a10c["gf"],
            "ga10_comp_diff": h10c["ga"] - a10c["ga"],
            "opp_elo10_overall_diff": h10o["opp_elo"] - a10o["opp_elo"],
            "hist10_overall_a": h10o["count"],
            "hist10_overall_b": a10o["count"],
            "hist10_comp_a": h10c["count"],
            "hist10_comp_b": a10c["count"],
            "b365_prob_a": imp_h,
            "b365_prob_draw": imp_d,
            "b365_prob_b": imp_a,
            "b365_prob_a_novig": nv_h,
            "b365_prob_draw_novig": nv_d,
            "b365_prob_b_novig": nv_a,
        }

        row_mirror = dict(common)
        row_mirror.update(
            {
                "team_a": away,
                "team_b": home,
                "team_a_is_home": 0,
                "elo_overall_diff": a_elo_o - h_elo_o,
                "elo_comp_diff": a_elo_c - h_elo_c,
                "rest_diff": a_rest - h_rest,
                "team_a_rest": a_rest,
                "team_b_rest": h_rest,
                "team_a_b2b": int(a_rest <= 3),
                "team_b_b2b": int(h_rest <= 3),
                "season_wr_comp_diff": a_wr - h_wr,
                "wr5_overall_diff": a5o["wr"] - h5o["wr"],
                "wr10_overall_diff": a10o["wr"] - h10o["wr"],
                "margin10_overall_diff": a10o["margin"] - h10o["margin"],
                "gf10_overall_diff": a10o["gf"] - h10o["gf"],
                "ga10_overall_diff": a10o["ga"] - h10o["ga"],
                "wr5_comp_diff": a5c["wr"] - h5c["wr"],
                "wr10_comp_diff": a10c["wr"] - h10c["wr"],
                "margin10_comp_diff": a10c["margin"] - h10c["margin"],
                "gf10_comp_diff": a10c["gf"] - h10c["gf"],
                "ga10_comp_diff": a10c["ga"] - h10c["ga"],
                "opp_elo10_overall_diff": a10o["opp_elo"] - h10o["opp_elo"],
                "hist10_overall_a": a10o["count"],
                "hist10_overall_b": h10o["count"],
                "hist10_comp_a": a10c["count"],
                "hist10_comp_b": h10c["count"],
                "b365_prob_a": imp_a,
                "b365_prob_b": imp_h,
                "b365_prob_a_novig": nv_a,
                "b365_prob_b_novig": nv_h,
            }
        )

        common["target_a_win"] = home_win
        row_mirror["target_a_win"] = away_win

        rows.append(common)
        rows.append(row_mirror)

        # Update states after feature row creation.
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

    out = pd.DataFrame(rows)
    out = out.sort_values(["match_date", "competition_code", "team_a", "team_b"]).reset_index(drop=True)
    return out


def split_by_recent_seasons(df: pd.DataFrame):
    seasons = sorted(df["season_start"].dropna().astype(int).unique())
    if len(seasons) < 6:
        raise ValueError("Need at least 6 seasons")
    test_seasons = seasons[-2:]
    val_seasons = [seasons[-3]]
    train_seasons = seasons[:-3]
    train = df[df["season_start"].isin(train_seasons)].copy()
    val = df[df["season_start"].isin(val_seasons)].copy()
    test = df[df["season_start"].isin(test_seasons)].copy()
    return train, val, test, train_seasons, val_seasons, test_seasons


def run_model(name, model, Xtr, ytr, Xva, yva, Xte, yte, use_eval=False):
    if use_eval:
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    else:
        model.fit(Xtr, ytr)
    pva = model.predict_proba(Xva)[:, 1]
    pte = model.predict_proba(Xte)[:, 1]

    out = {
        "Model": name,
        "ValAcc": accuracy_score(yva, (pva >= 0.5).astype(int)),
        "TestAcc": accuracy_score(yte, (pte >= 0.5).astype(int)),
        "ValAUC": roc_auc_score(yva, pva),
        "TestAUC": roc_auc_score(yte, pte),
        "ValLL": log_loss(yva, pva),
        "TestLL": log_loss(yte, pte),
        "ValBrier": brier_score_loss(yva, pva),
        "TestBrier": brier_score_loss(yte, pte),
        "ValECE": ece_score(yva, pva),
        "TestECE": ece_score(yte, pte),
    }
    return out, pva, pte, model


def main():
    parser = argparse.ArgumentParser(description="European club soccer model pipeline")
    parser.add_argument("--data-dir", default="Soccer\\soccer_data")
    parser.add_argument("--outdir", default="Soccer\\soccer_outputs")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    matches_path = os.path.join(args.data_dir, "matches.csv")
    if not os.path.exists(matches_path):
        raise FileNotFoundError(f"Missing {matches_path}. Run soccer_scraper.py first.")

    matches = pd.read_csv(matches_path)
    matches["match_date"] = pd.to_datetime(matches["match_date"], errors="coerce")
    matches = matches.dropna(subset=["match_date", "home_goals", "away_goals"])
    matches = matches.sort_values(["match_date", "competition_code", "home_team", "away_team"]).reset_index(drop=True)

    print(f"[DATA] matches: {len(matches):,}")

    feat = build_feature_table(matches)
    feat_path = os.path.join(args.outdir, "soccer_feature_table.csv")
    feat.to_csv(feat_path, index=False)
    print(f"[SAVE] {feat_path}")
    print(f"[DATA] mirrored rows: {len(feat):,}")

    train, val, test, tr_s, va_s, te_s = split_by_recent_seasons(feat)
    print(f"[SPLIT] train seasons: {tr_s[0]}..{tr_s[-1]} ({len(train):,})")
    print(f"[SPLIT] val seasons:   {va_s} ({len(val):,})")
    print(f"[SPLIT] test seasons:  {te_s} ({len(test):,})")

    drop_cols = {"target_a_win", "match_date"}
    fcols = [c for c in feat.columns if c not in drop_cols]

    Xtr = train[fcols].copy()
    ytr = train["target_a_win"].astype(int).values
    Xva = val[fcols].copy()
    yva = val["target_a_win"].astype(int).values
    Xte = test[fcols].copy()
    yte = test["target_a_win"].astype(int).values

    # Encode team + competition categoricals.
    cat_cols = ["competition_code", "competition_name", "team_a", "team_b"]
    allX = pd.concat([Xtr, Xva, Xte], axis=0)
    allE = pd.get_dummies(allX, columns=cat_cols, dummy_na=False)

    XtrE = allE.iloc[: len(Xtr)].copy()
    XvaE = allE.iloc[len(Xtr) : len(Xtr) + len(Xva)].copy()
    XteE = allE.iloc[len(Xtr) + len(Xva) :].copy()

    imp = SimpleImputer(strategy="median")
    XtrI = imp.fit_transform(XtrE)
    XvaI = imp.transform(XvaE)
    XteI = imp.transform(XteE)

    rows = []

    dt = DecisionTreeClassifier(max_depth=12, min_samples_leaf=25, random_state=42)
    r, _, _, _ = run_model("Decision Tree", dt, XtrI, ytr, XvaI, yva, XteI, yte)
    rows.append(r)

    rf = RandomForestClassifier(n_estimators=700, max_depth=16, min_samples_leaf=8, n_jobs=-1, random_state=42)
    r, _, _, _ = run_model("Random Forest", rf, XtrI, ytr, XvaI, yva, XteI, yte)
    rows.append(r)

    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=8,
        reg_alpha=0.2,
        reg_lambda=2.0,
        eval_metric="logloss",
        early_stopping_rounds=40,
        random_state=42,
        verbosity=0,
    )
    r, pva_xgb, pte_xgb, fitted_xgb = run_model("XGBoost", xgb_model, XtrI, ytr, XvaI, yva, XteI, yte, use_eval=True)
    rows.append(r)

    nn = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=350,
        random_state=42,
    )
    r, _, _, _ = run_model("Neural Network", nn, XtrI, ytr, XvaI, yva, XteI, yte)
    rows.append(r)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(pva_xgb, yva)
    pva_iso = iso.transform(pva_xgb)
    pte_iso = iso.transform(pte_xgb)
    rows.append(
        {
            "Model": "XGBoost + Isotonic",
            "ValAcc": accuracy_score(yva, (pva_iso >= 0.5).astype(int)),
            "TestAcc": accuracy_score(yte, (pte_iso >= 0.5).astype(int)),
            "ValAUC": roc_auc_score(yva, pva_iso),
            "TestAUC": roc_auc_score(yte, pte_iso),
            "ValLL": log_loss(yva, pva_iso),
            "TestLL": log_loss(yte, pte_iso),
            "ValBrier": brier_score_loss(yva, pva_iso),
            "TestBrier": brier_score_loss(yte, pte_iso),
            "ValECE": ece_score(yva, pva_iso),
            "TestECE": ece_score(yte, pte_iso),
        }
    )

    rep = pd.DataFrame(rows)
    rep_path = os.path.join(args.outdir, "soccer_model_report.csv")
    rep.to_csv(rep_path, index=False)

    joblib.dump(fitted_xgb, os.path.join(args.outdir, "soccer_xgb_model.pkl"))
    joblib.dump(imp, os.path.join(args.outdir, "soccer_imputer.pkl"))
    joblib.dump(iso, os.path.join(args.outdir, "soccer_isotonic.pkl"))

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_cols": fcols,
        "encoded_columns": list(XtrE.columns),
        "train_seasons": [int(x) for x in tr_s],
        "val_seasons": [int(x) for x in va_s],
        "test_seasons": [int(x) for x in te_s],
        "split_rows": {"train": int(len(train)), "val": int(len(val)), "test": int(len(test))},
        "mirrored_training": True,
        "scope": "European club competitions only",
    }
    with open(os.path.join(args.outdir, "soccer_artifacts_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n[REPORT]")
    print(rep.to_string(index=False))
    print(f"\n[SAVE] {rep_path}")


if __name__ == "__main__":
    main()
