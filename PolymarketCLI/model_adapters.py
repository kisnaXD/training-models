#!/usr/bin/env python3
"""Adapters that bridge Polymarket markets to local sports EV model CLIs."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _load_module(name: str, path: Path):
    mod_dir = str(path.parent.resolve())
    added = False
    if mod_dir not in sys.path:
        sys.path.insert(0, mod_dir)
        added = True
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        if added:
            try:
                sys.path.remove(mod_dir)
            except ValueError:
                pass
    return module


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


class NBAAdapter:
    def __init__(self):
        self.mod = _load_module("nba_ev_cli_mod", ROOT / "NBA" / "nba_ev_cli.py")
        teams = pd.read_csv(ROOT / "NBA" / "nba_data" / "teams.csv")
        name_to_abbr: Dict[str, str] = {}
        for _, r in teams.iterrows():
            abbr = str(r["team_abbreviation"]).upper()
            full = str(r["team_full_name"])
            city = str(r["team_city"])
            nick = str(r["team_nickname"])
            cands = {abbr, full, nick, f"{city} {nick}"}
            for c in cands:
                name_to_abbr[_norm(c)] = abbr
        name_to_abbr[_norm("76ers")] = "PHI"
        self.name_to_abbr = name_to_abbr

    def team_to_code(self, name: str) -> Optional[str]:
        return self.name_to_abbr.get(_norm(name))

    def run(self, markets: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for _, r in markets.iterrows():
            a_code = self.team_to_code(r["outcome_a"])
            b_code = self.team_to_code(r["outcome_b"])
            rows.append(
                {
                    "team_a": a_code if a_code else r["outcome_a"],
                    "team_b": b_code if b_code else r["outcome_b"],
                    "game_date": pd.to_datetime(r["start_date"], errors="coerce").date() if pd.notna(r["start_date"]) else pd.Timestamp.today().date(),
                    "price_a_cents": r["price_a_cents"],
                    "price_b_cents": r["price_b_cents"],
                    "orig_outcome_a": r["outcome_a"],
                    "orig_outcome_b": r["outcome_b"],
                }
            )
        df = pd.DataFrame(rows)
        invoke_df = df[["team_a", "team_b", "game_date", "price_a_cents", "price_b_cents"]].copy()
        tmp = ROOT / "NBA" / "nba_outputs" / "nba_ev_cli_polymarket_tmp.csv"
        with contextlib.redirect_stdout(io.StringIO()):
            self.mod.run(invoke_df, str(ROOT / "NBA"), output_path=str(tmp))
        out = pd.read_csv(tmp)
        out["orig_outcome_a"] = df["orig_outcome_a"].values
        out["orig_outcome_b"] = df["orig_outcome_b"].values
        return out


class NFLAdapter:
    TEAM_MAP = {
        "cardinals": "ARI",
        "falcons": "ATL",
        "ravens": "BAL",
        "bills": "BUF",
        "panthers": "CAR",
        "bears": "CHI",
        "bengals": "CIN",
        "browns": "CLE",
        "cowboys": "DAL",
        "broncos": "DEN",
        "lions": "DET",
        "packers": "GB",
        "texans": "HOU",
        "colts": "IND",
        "jaguars": "JAX",
        "chiefs": "KC",
        "rams": "LA",
        "chargers": "LAC",
        "raiders": "LV",
        "dolphins": "MIA",
        "vikings": "MIN",
        "patriots": "NE",
        "saints": "NO",
        "giants": "NYG",
        "jets": "NYJ",
        "eagles": "PHI",
        "steelers": "PIT",
        "seahawks": "SEA",
        "49ers": "SF",
        "niners": "SF",
        "buccaneers": "TB",
        "bucs": "TB",
        "titans": "TEN",
        "commanders": "WAS",
        "washington": "WAS",
    }

    def __init__(self):
        self.mod = _load_module("nfl_ev_cli_mod", ROOT / "NFL" / "nfl_ev_cli.py")

    def team_to_code(self, name: str) -> Optional[str]:
        n = str(name).lower()
        for k, v in self.TEAM_MAP.items():
            if k in n:
                return v
        s = str(name).upper().strip()
        if len(s) <= 3:
            return s
        return None

    def run(self, markets: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for _, r in markets.iterrows():
            home = self.team_to_code(r["outcome_a"])
            away = self.team_to_code(r["outcome_b"])
            rows.append(
                {
                    "home_team": home if home else r["outcome_a"],
                    "away_team": away if away else r["outcome_b"],
                    "game_date": pd.to_datetime(r["start_date"], errors="coerce").date() if pd.notna(r["start_date"]) else pd.Timestamp.today().date(),
                    "price_home_cents": r["price_a_cents"],
                    "price_away_cents": r["price_b_cents"],
                    "orig_outcome_a": r["outcome_a"],
                    "orig_outcome_b": r["outcome_b"],
                }
            )
        df = pd.DataFrame(rows)
        invoke_df = df[["home_team", "away_team", "game_date", "price_home_cents", "price_away_cents"]].copy()
        tmp = ROOT / "NFL" / "nfl_outputs" / "nfl_ev_cli_polymarket_tmp.csv"
        with contextlib.redirect_stdout(io.StringIO()):
            self.mod.run_report(invoke_df, str(ROOT / "NFL"), output_csv=str(tmp))
        out = pd.read_csv(tmp)
        out["orig_outcome_a"] = df["orig_outcome_a"].values
        out["orig_outcome_b"] = df["orig_outcome_b"].values
        return out


class UFCAdapter:
    def __init__(self):
        self.mod = _load_module("ufc_ev_cli_mod", ROOT / "UFC" / "ufc_ev_cli.py")

    def run(self, markets: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for _, r in markets.iterrows():
            rows.append(
                {
                    "fighter_a": r["outcome_a"],
                    "fighter_b": r["outcome_b"],
                    "fight_date": pd.to_datetime(r["start_date"], errors="coerce").date() if pd.notna(r["start_date"]) else pd.Timestamp.today().date(),
                    "weight_class": "Lightweight",
                    "price_a_cents": r["price_a_cents"],
                    "price_b_cents": r["price_b_cents"],
                }
            )
        df = pd.DataFrame(rows)
        tmp = ROOT / "UFC" / "ufc_ev_cli_polymarket_tmp.csv"
        with contextlib.redirect_stdout(io.StringIO()):
            self.mod.run(df, str(ROOT / "UFC"), output_path=str(tmp))
        out = pd.read_csv(tmp)
        return out.copy()


class SoccerAdapter:
    def __init__(self):
        self.mod = _load_module("soccer_ev_cli_3way_mod", ROOT / "Soccer" / "soccer_ev_cli_3way.py")

    def run(self, markets: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for _, r in markets.iterrows():
            rows.append(
                {
                    "team_a": r["outcome_a"],
                    "team_b": r["outcome_b"],
                    "title": r.get("title", ""),
                    "game_date": pd.to_datetime(r["start_date"], errors="coerce").date() if pd.notna(r["start_date"]) else pd.Timestamp.today().date(),
                    "price_a_cents": r["price_a_cents"],
                    "price_draw_cents": r.get("price_draw_cents", np.nan),
                    "price_b_cents": r["price_b_cents"],
                    "orig_outcome_a": r["outcome_a"],
                    "orig_outcome_b": r["outcome_b"],
                    "orig_outcome_draw": r.get("outcome_draw", ""),
                }
            )
        df = pd.DataFrame(rows)
        tmp = ROOT / "Soccer" / "soccer_outputs_3way" / "soccer_ev_cli_polymarket_tmp.csv"
        with contextlib.redirect_stdout(io.StringIO()):
            self.mod.run(df, str(ROOT / "Soccer"), output_path=str(tmp))
        out = pd.read_csv(tmp)
        out["orig_outcome_a"] = df["orig_outcome_a"].values
        out["orig_outcome_b"] = df["orig_outcome_b"].values
        if "orig_outcome_draw" in df.columns:
            out["orig_outcome_draw"] = df["orig_outcome_draw"].values
        return out.copy()


def predict_with_model(sport: str, markets: pd.DataFrame) -> pd.DataFrame:
    sport_key = str(sport).strip().lower()
    if sport_key == "nba":
        return NBAAdapter().run(markets)
    if sport_key == "nfl":
        return NFLAdapter().run(markets)
    if sport_key == "ufc":
        return UFCAdapter().run(markets)
    if sport_key == "soccer":
        return SoccerAdapter().run(markets)
    raise ValueError(f"Model adapter not available for sport '{sport}'. Supported: nba, nfl, ufc, soccer.")
