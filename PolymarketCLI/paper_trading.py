#!/usr/bin/env python3
"""Paper trading helpers for Polymarket sports CLI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

PAPER_DIR = Path("PolymarketCLI") / "paper"
LEDGER_CSV = PAPER_DIR / "paper_ledger.csv"


@dataclass
class PaperConfig:
    default_stake_usd: float = 25.0


def _ensure_dir():
    PAPER_DIR.mkdir(parents=True, exist_ok=True)


def load_ledger() -> pd.DataFrame:
    _ensure_dir()
    if LEDGER_CSV.exists():
        return pd.read_csv(LEDGER_CSV)
    cols = [
        "ts",
        "sport",
        "event_id",
        "market_id",
        "title",
        "side_name",
        "side_col",
        "entry_price_cents",
        "shares",
        "stake_usd",
        "model_prob",
        "model_edge_cents",
        "action",
        "status",
        "exit_price_cents",
        "realized_pnl_usd",
    ]
    return pd.DataFrame(columns=cols)


def save_ledger(df: pd.DataFrame):
    _ensure_dir()
    df.to_csv(LEDGER_CSV, index=False)


def append_positions(df_new: pd.DataFrame):
    if df_new.empty:
        return
    led = load_ledger()
    led = pd.concat([led, df_new], ignore_index=True)
    save_ledger(led)


def build_new_positions_from_reco(
    merged_df: pd.DataFrame,
    stake_usd: float,
    only_buy: bool = True,
) -> pd.DataFrame:
    def _pick(row, key, default=None):
        if key not in row:
            return default
        v = row[key]
        if isinstance(v, pd.Series):
            if len(v) == 0:
                return default
            return v.iloc[0]
        return v

    rows = []
    now = datetime.now().isoformat(timespec="seconds")
    for _, r in merged_df.iterrows():
        action = str(_pick(r, "action", "")).strip().upper()
        if only_buy and action != "BUY":
            continue

        # infer chosen side and corresponding price/prob columns
        best_side = str(_pick(r, "best_side", "")).strip()
        out_a = str(_pick(r, "outcome_a", "")).strip()
        out_b = str(_pick(r, "outcome_b", "")).strip()

        price_a = float(_pick(r, "price_a_cents", _pick(r, "price_home_cents", 0.0)) or 0.0)
        price_b = float(_pick(r, "price_b_cents", _pick(r, "price_away_cents", 0.0)) or 0.0)
        prob_a = _pick(r, "prob_a_cal", _pick(r, "prob_home_cal", None))
        prob_b = _pick(r, "prob_b_cal", _pick(r, "prob_away_cal", None))
        edge_a = _pick(r, "edge_a_cents", _pick(r, "edge_home_cents", None))
        edge_b = _pick(r, "edge_b_cents", _pick(r, "edge_away_cents", None))

        side_col = "A"
        side_name = out_a
        entry_price = price_a
        model_prob = prob_a
        model_edge = edge_a

        if best_side and (best_side.lower() == out_b.lower()):
            side_col = "B"
            side_name = out_b
            entry_price = price_b
            model_prob = prob_b
            model_edge = edge_b
        elif best_side and (best_side.lower() == out_a.lower()):
            pass
        else:
            # Fallback: choose side with higher edge if available.
            if edge_b is not None and edge_a is not None and float(edge_b) > float(edge_a):
                side_col = "B"
                side_name = out_b
                entry_price = price_b
                model_prob = prob_b
                model_edge = edge_b

        if entry_price <= 0:
            continue
        shares = stake_usd / (entry_price / 100.0)

        rows.append(
            {
                "ts": now,
                "sport": _pick(r, "sport", ""),
                "event_id": _pick(r, "event_id", ""),
                "market_id": _pick(r, "market_id", ""),
                "title": _pick(r, "title", ""),
                "side_name": side_name,
                "side_col": side_col,
                "entry_price_cents": round(entry_price, 4),
                "shares": round(shares, 6),
                "stake_usd": round(stake_usd, 6),
                "model_prob": model_prob,
                "model_edge_cents": model_edge,
                "action": action,
                "status": "OPEN",
                "exit_price_cents": None,
                "realized_pnl_usd": None,
            }
        )
    return pd.DataFrame(rows)


def mark_to_market(ledger_df: pd.DataFrame, latest_df: pd.DataFrame) -> pd.DataFrame:
    if ledger_df.empty:
        return ledger_df.copy()
    out = ledger_df.copy()
    out["current_price_cents"] = pd.NA
    out["market_value_usd"] = pd.NA
    out["unrealized_pnl_usd"] = pd.NA

    idx_by_key: Dict[str, pd.Series] = {}
    for _, r in latest_df.iterrows():
        key = f'{r.get("event_id","")}::{r.get("market_id","")}'
        idx_by_key[key] = r

    for i, r in out.iterrows():
        key = f'{r.get("event_id","")}::{r.get("market_id","")}'
        live = idx_by_key.get(key)
        if live is None:
            continue
        side_col = str(r.get("side_col", "A")).upper()
        p = float(live.get("price_a_cents", 0.0)) if side_col == "A" else float(live.get("price_b_cents", 0.0))
        shares = float(r.get("shares", 0.0))
        stake = float(r.get("stake_usd", 0.0))
        mv = shares * (p / 100.0)
        out.at[i, "current_price_cents"] = round(p, 4)
        out.at[i, "market_value_usd"] = round(mv, 6)
        out.at[i, "unrealized_pnl_usd"] = round(mv - stake, 6)
    return out


def summarize_portfolio(mtm_df: pd.DataFrame) -> Dict[str, float]:
    if mtm_df.empty:
        return {
            "positions_open": 0,
            "total_stake_usd": 0.0,
            "market_value_usd": 0.0,
            "unrealized_pnl_usd": 0.0,
        }
    open_df = mtm_df[mtm_df["status"].astype(str).str.upper() == "OPEN"].copy()
    total_stake = float(open_df["stake_usd"].fillna(0).sum())
    mv = float(pd.to_numeric(open_df["market_value_usd"], errors="coerce").fillna(0).sum())
    upnl = float(pd.to_numeric(open_df["unrealized_pnl_usd"], errors="coerce").fillna(0).sum())
    return {
        "positions_open": int(len(open_df)),
        "total_stake_usd": total_stake,
        "market_value_usd": mv,
        "unrealized_pnl_usd": upnl,
    }
