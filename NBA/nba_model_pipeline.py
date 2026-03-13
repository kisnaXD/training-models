#!/usr/bin/env python3
"""
NBA Team-Sport Modeling Pipeline (NBA only)

Builds leakage-safe pre-game features from raw team logs, trains multiple models,
calibrates probabilities, and saves artifacts.

Usage:
  python nba\nba_model_pipeline.py --data-dir nba_data --outdir nba_outputs
"""

import argparse
import json
import os
from collections import defaultdict, deque

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.isotonic import IsotonicRegression

import xgboost as xgb


np.random.seed(42)
ELO_BASE = 1500.0
HOME_ADV = 80.0


def parse_season_start(season_label: str) -> int:
    # "2024-25" -> 2024
    try:
        return int(str(season_label).split("-")[0])
    except Exception:
        return -1


def parse_matchup(matchup: str):
    txt = str(matchup)
    if " vs. " in txt:
        left, right = txt.split(" vs. ")
        return True, right.strip()
    if " @ " in txt:
        left, right = txt.split(" @ ")
        return False, right.strip()
    return False, ""


def ece_score(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bucket = np.digitize(y_prob, bins, right=True)
    n = len(y_true)
    ece = 0.0
    for b in range(1, n_bins + 1):
        idx = bucket == b
        if not np.any(idx):
            continue
        conf = np.mean(y_prob[idx])
        acc = np.mean(y_true[idx])
        ece += (np.sum(idx) / n) * abs(acc - conf)
    return float(ece)


def elo_expect(home_elo, away_elo):
    return 1.0 / (1.0 + 10.0 ** ((away_elo - home_elo) / 400.0))


def elo_update(home_elo, away_elo, home_win, mov, playoffs=False):
    # MOV-aware Elo update for team sports.
    expected_home = elo_expect(home_elo + HOME_ADV, away_elo)
    actual_home = 1.0 if home_win else 0.0
    elo_gap = (home_elo + HOME_ADV) - away_elo
    mov_mult = ((abs(mov) + 3.0) ** 0.8) / (7.5 + 0.006 * abs(elo_gap))
    k = 24.0 if playoffs else 20.0
    delta = k * mov_mult * (actual_home - expected_home)
    return home_elo + delta, away_elo - delta


def summarize_hist(hist, n=None):
    h = list(hist)[-n:] if n else list(hist)
    total = len(h)
    if total == 0:
        return {
            "win_rate": 0.5,
            "avg_pts": 110.0,
            "avg_opp_pts": 110.0,
            "net": 0.0,
            "avg_efg": 0.52,
            "avg_tov": 14.0,
            "avg_reb": 44.0,
            "avg_ast": 24.0,
            "avg_opp_elo": ELO_BASE,
            "count": 0,
        }
    wins = sum(x["win"] for x in h)
    return {
        "win_rate": wins / total,
        "avg_pts": float(np.mean([x["pts"] for x in h])),
        "avg_opp_pts": float(np.mean([x["opp_pts"] for x in h])),
        "net": float(np.mean([x["pts"] - x["opp_pts"] for x in h])),
        "avg_efg": float(np.mean([x["efg"] for x in h])),
        "avg_tov": float(np.mean([x["tov"] for x in h])),
        "avg_reb": float(np.mean([x["reb"] for x in h])),
        "avg_ast": float(np.mean([x["ast"] for x in h])),
        "avg_opp_elo": float(np.mean([x.get("opp_elo", ELO_BASE) for x in h])),
        "count": total,
    }


def load_franchise_priors(franchise_history: pd.DataFrame):
    priors = {}
    if len(franchise_history) == 0:
        return priors

    fh = franchise_history.copy()
    fh["season_start"] = fh["season_label"].astype(str).apply(parse_season_start)
    for _, r in fh.iterrows():
        tid = int(r["team_id"])
        season = int(r["season_start"])
        priors[(tid, season)] = {
            "prior_win_pct": float(r.get("win_pct", np.nan)) if pd.notna(r.get("win_pct", np.nan)) else 0.5,
            "prior_playoff_wins": float(r.get("playoff_wins", 0) if pd.notna(r.get("playoff_wins", np.nan)) else 0),
            "prior_playoff_losses": float(r.get("playoff_losses", 0) if pd.notna(r.get("playoff_losses", np.nan)) else 0),
        }
    return priors


def build_game_rows(raw_logs: pd.DataFrame):
    logs = raw_logs.copy()
    parsed = logs["matchup"].apply(parse_matchup)
    logs["is_home"] = parsed.apply(lambda x: x[0])
    logs["opp_abbr"] = parsed.apply(lambda x: x[1])
    logs["game_date"] = pd.to_datetime(logs["game_date"], errors="coerce")

    out = []
    for game_id, g in logs.groupby("game_id", sort=False):
        if len(g) != 2:
            continue
        g = g.sort_values("is_home")
        away = g.iloc[0]
        home = g.iloc[1]
        out.append(
            {
                "game_id": game_id,
                "game_date": home["game_date"],
                "season": home["season"],
                "season_type": home["season_type"],
                "home_team_id": int(home["team_id"]),
                "away_team_id": int(away["team_id"]),
                "home_team_abbr": home["team_abbreviation"],
                "away_team_abbr": away["team_abbreviation"],
                "home_pts": float(home["pts"]),
                "away_pts": float(away["pts"]),
                "home_wl": str(home["wl"]),
                "away_wl": str(away["wl"]),
                "home_fga": float(home["fga"]),
                "away_fga": float(away["fga"]),
                "home_fg3m": float(home["fg3m"]),
                "away_fg3m": float(away["fg3m"]),
                "home_fgm": float(home["fgm"]),
                "away_fgm": float(away["fgm"]),
                "home_fta": float(home["fta"]),
                "away_fta": float(away["fta"]),
                "home_tov": float(home["tov"]),
                "away_tov": float(away["tov"]),
                "home_reb": float(home["reb"]),
                "away_reb": float(away["reb"]),
                "home_ast": float(home["ast"]),
                "away_ast": float(away["ast"]),
            }
        )
    out_df = pd.DataFrame(out)
    out_df = out_df.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    return out_df


def build_feature_table(games: pd.DataFrame, priors: dict):
    elo = defaultdict(lambda: ELO_BASE)
    team_hist = defaultdict(lambda: deque(maxlen=120))
    last_game_date = {}
    season_record = defaultdict(lambda: {"w": 0, "l": 0})

    current_season = None
    rows = []

    for _, g in games.iterrows():
        game_date = g["game_date"]
        season = str(g["season"])
        season_start = parse_season_start(season)

        if current_season is None:
            current_season = season
        if season != current_season:
            # offseason mean-reversion on Elo + reset season records
            for tid in list(elo.keys()):
                elo[tid] = 0.75 * elo[tid] + 0.25 * ELO_BASE
            season_record = defaultdict(lambda: {"w": 0, "l": 0})
            current_season = season

        home = int(g["home_team_id"])
        away = int(g["away_team_id"])
        home_win = int(g["home_pts"] > g["away_pts"])
        mov = float(g["home_pts"] - g["away_pts"])
        playoffs = str(g["season_type"]) == "Playoffs"

        home_elo = elo[home]
        away_elo = elo[away]

        # rest / fatigue
        home_rest = (game_date - last_game_date[home]).days if home in last_game_date else 5
        away_rest = (game_date - last_game_date[away]).days if away in last_game_date else 5
        home_b2b = int(home_rest <= 1)
        away_b2b = int(away_rest <= 1)

        # historical windows
        h5 = summarize_hist(team_hist[home], 5)
        h10 = summarize_hist(team_hist[home], 10)
        h20 = summarize_hist(team_hist[home], 20)
        a5 = summarize_hist(team_hist[away], 5)
        a10 = summarize_hist(team_hist[away], 10)
        a20 = summarize_hist(team_hist[away], 20)

        # season record prior to game
        h_rec = season_record[(home, season)]
        a_rec = season_record[(away, season)]
        h_games = h_rec["w"] + h_rec["l"]
        a_games = a_rec["w"] + a_rec["l"]
        h_season_wr = (h_rec["w"] / h_games) if h_games > 0 else 0.5
        a_season_wr = (a_rec["w"] / a_games) if a_games > 0 else 0.5

        # franchise priors from previous season
        home_prior = priors.get((home, season_start - 1), {"prior_win_pct": 0.5, "prior_playoff_wins": 0.0, "prior_playoff_losses": 0.0})
        away_prior = priors.get((away, season_start - 1), {"prior_win_pct": 0.5, "prior_playoff_wins": 0.0, "prior_playoff_losses": 0.0})

        feat = {
            "season": season,
            "season_start": season_start,
            "game_date": game_date,
            "game_id": g["game_id"],
            "home_team_id": home,
            "away_team_id": away,
            "home_team_abbr": g["home_team_abbr"],
            "away_team_abbr": g["away_team_abbr"],
            "playoffs": int(playoffs),

            "elo_diff": home_elo - away_elo,
            "home_elo": home_elo,
            "away_elo": away_elo,

            "rest_diff": home_rest - away_rest,
            "home_rest": home_rest,
            "away_rest": away_rest,
            "home_b2b": home_b2b,
            "away_b2b": away_b2b,

            "season_wr_diff": h_season_wr - a_season_wr,
            "home_season_wr": h_season_wr,
            "away_season_wr": a_season_wr,
            "home_season_games": h_games,
            "away_season_games": a_games,

            "prior_win_pct_diff": home_prior["prior_win_pct"] - away_prior["prior_win_pct"],
            "home_prior_win_pct": home_prior["prior_win_pct"],
            "away_prior_win_pct": away_prior["prior_win_pct"],
            "prior_playoff_wins_diff": home_prior["prior_playoff_wins"] - away_prior["prior_playoff_wins"],

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

            "target_home_win": home_win,
            "home_pts": g["home_pts"],
            "away_pts": g["away_pts"],
        }
        rows.append(feat)

        # update per-team game summaries AFTER feature row is logged
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

        # season records + last played date
        season_record[(home, season)]["w"] += home_win
        season_record[(home, season)]["l"] += (1 - home_win)
        season_record[(away, season)]["w"] += (1 - home_win)
        season_record[(away, season)]["l"] += home_win
        last_game_date[home] = game_date
        last_game_date[away] = game_date

        # elo update
        new_home, new_away = elo_update(home_elo, away_elo, bool(home_win), mov, playoffs=playoffs)
        elo[home] = new_home
        elo[away] = new_away

    return pd.DataFrame(rows)


def split_by_recent_seasons(df: pd.DataFrame):
    seasons = sorted(df["season"].dropna().unique(), key=parse_season_start)
    if len(seasons) < 6:
        raise ValueError("Need at least 6 seasons for robust split")

    test_seasons = seasons[-2:]
    val_seasons = [seasons[-3]]
    train_seasons = seasons[:-3]

    train_df = df[df["season"].isin(train_seasons)].copy()
    val_df = df[df["season"].isin(val_seasons)].copy()
    test_df = df[df["season"].isin(test_seasons)].copy()
    return train_df, val_df, test_df, train_seasons, val_seasons, test_seasons


def evaluate_probs(name, y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "logloss": float(log_loss(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": float(ece_score(y_true, y_prob)),
    }


def main():
    parser = argparse.ArgumentParser(description="NBA feature engineering + training pipeline")
    parser.add_argument("--data-dir", type=str, default="nba_data", help="Folder containing raw_team_game_logs.csv and franchise_history.csv")
    parser.add_argument("--outdir", type=str, default="nba_outputs", help="Output folder for features/models/reports")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    raw_path = os.path.join(args.data_dir, "raw_team_game_logs.csv")
    fh_path = os.path.join(args.data_dir, "franchise_history.csv")

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing file: {raw_path}")
    if not os.path.exists(fh_path):
        raise FileNotFoundError(f"Missing file: {fh_path}")

    print("=" * 72)
    print("NBA MODEL PIPELINE")
    print("=" * 72)

    raw_logs = pd.read_csv(raw_path)
    franchise_history = pd.read_csv(fh_path)

    print(f"[DATA] raw team-game rows: {len(raw_logs):,}")

    games = build_game_rows(raw_logs)
    priors = load_franchise_priors(franchise_history)

    print(f"[DATA] merged games: {len(games):,}")

    feat_df = build_feature_table(games, priors)
    feat_df = feat_df.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    feature_path = os.path.join(args.outdir, "nba_feature_table.csv")
    feat_df.to_csv(feature_path, index=False)
    print(f"[SAVE] {feature_path}")

    train_df, val_df, test_df, tr_s, va_s, te_s = split_by_recent_seasons(feat_df)
    print(f"[SPLIT] train seasons: {tr_s[0]} .. {tr_s[-1]} ({len(train_df):,} rows)")
    print(f"[SPLIT] val seasons:   {va_s} ({len(val_df):,} rows)")
    print(f"[SPLIT] test seasons:  {te_s} ({len(test_df):,} rows)")

    drop_cols = {
        "target_home_win",
        "game_date",
        "game_id",
        "season",
        "home_team_abbr",
        "away_team_abbr",
        "home_pts",
        "away_pts",
    }
    feature_cols = [c for c in feat_df.columns if c not in drop_cols]

    X_train = train_df[feature_cols].copy()
    y_train = train_df["target_home_win"].values
    X_val = val_df[feature_cols].copy()
    y_val = val_df["target_home_win"].values
    X_test = test_df[feature_cols].copy()
    y_test = test_df["target_home_win"].values

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)

    # 1) Logistic baseline
    logit = LogisticRegression(max_iter=2000, solver="lbfgs")
    logit.fit(X_train_imp, y_train)
    p_logit_val = logit.predict_proba(X_val_imp)[:, 1]
    p_logit_test = logit.predict_proba(X_test_imp)[:, 1]

    # 2) Random forest
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train_imp, y_train)
    p_rf_val = rf.predict_proba(X_val_imp)[:, 1]
    p_rf_test = rf.predict_proba(X_test_imp)[:, 1]

    # 3) XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=900,
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
    xgb_model.fit(X_train_imp, y_train, eval_set=[(X_val_imp, y_val)], verbose=False)
    p_xgb_val_raw = xgb_model.predict_proba(X_val_imp)[:, 1]
    p_xgb_test_raw = xgb_model.predict_proba(X_test_imp)[:, 1]

    # Isotonic calibration using validation predictions
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_xgb_val_raw, y_val)
    p_xgb_val_cal = iso.transform(p_xgb_val_raw)
    p_xgb_test_cal = iso.transform(p_xgb_test_raw)

    reports = []
    reports.append(evaluate_probs("Logistic", y_val, p_logit_val) | {"split": "val"})
    reports.append(evaluate_probs("RandomForest", y_val, p_rf_val) | {"split": "val"})
    reports.append(evaluate_probs("XGBoost", y_val, p_xgb_val_raw) | {"split": "val"})
    reports.append(evaluate_probs("XGBoost+Isotonic", y_val, p_xgb_val_cal) | {"split": "val"})

    reports.append(evaluate_probs("Logistic", y_test, p_logit_test) | {"split": "test"})
    reports.append(evaluate_probs("RandomForest", y_test, p_rf_test) | {"split": "test"})
    reports.append(evaluate_probs("XGBoost", y_test, p_xgb_test_raw) | {"split": "test"})
    reports.append(evaluate_probs("XGBoost+Isotonic", y_test, p_xgb_test_cal) | {"split": "test"})

    report_df = pd.DataFrame(reports)
    report_path = os.path.join(args.outdir, "nba_model_report.csv")
    report_df.to_csv(report_path, index=False)
    print(f"[SAVE] {report_path}")
    print("\n[REPORT]")
    print(report_df.to_string(index=False))

    # Save best stack (raw + calibrator so inference can choose/calibrate)
    joblib.dump(xgb_model, os.path.join(args.outdir, "nba_xgb_model.pkl"))
    joblib.dump(imputer, os.path.join(args.outdir, "nba_imputer.pkl"))
    joblib.dump(iso, os.path.join(args.outdir, "nba_isotonic.pkl"))

    artifacts = {
        "feature_cols": feature_cols,
        "train_seasons": tr_s,
        "val_seasons": va_s,
        "test_seasons": te_s,
        "split_rows": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
    }

    with open(os.path.join(args.outdir, "nba_artifacts_meta.json"), "w", encoding="utf-8") as f:
        json.dump(artifacts, f, indent=2)

    print("[SAVE] nba_xgb_model.pkl")
    print("[SAVE] nba_imputer.pkl")
    print("[SAVE] nba_isotonic.pkl")
    print("[SAVE] nba_artifacts_meta.json")

    print("=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
