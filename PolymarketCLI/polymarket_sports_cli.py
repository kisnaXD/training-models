#!/usr/bin/env python3
"""Polymarket Sports Terminal: deep market search + local model EV overlay."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from polymarket_api import GammaClient, SPORT_TAG_SLUGS
from model_adapters import predict_with_model
from paper_trading import (
    PaperConfig,
    append_positions,
    build_new_positions_from_reco,
    load_ledger,
    mark_to_market,
    summarize_portfolio,
)

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt
    from rich.table import Table
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Missing dependency 'rich'. Install with: pip install rich\nError: {exc}")


console = Console()


def _fmt_dt(s: str) -> str:
    if not s:
        return "-"
    try:
        return pd.Timestamp(s).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(s)


def _build_market_df(rows) -> pd.DataFrame:
    return pd.DataFrame([r.as_dict() for r in rows]) if rows else pd.DataFrame()


def render_market_table(df: pd.DataFrame, max_rows: int = 25):
    t = Table(title=f"Polymarket Sports Markets ({len(df)})", box=box.SIMPLE_HEAVY, show_lines=False)
    cols = [
        ("#", "right"),
        ("Sport", "left"),
        ("Event", "left"),
        ("A", "left"),
        ("Draw", "left"),
        ("B", "left"),
        ("A Price", "right"),
        ("D Price", "right"),
        ("B Price", "right"),
        ("Liquidity", "right"),
        ("Start", "left"),
    ]
    for c, j in cols:
        t.add_column(c, justify=j, overflow="fold")
    for i, (_, r) in enumerate(df.head(max_rows).iterrows(), 1):
        t.add_row(
            str(i),
            str(r["sport"]).upper(),
            str(r["title"])[:62],
            str(r["outcome_a"])[:22],
            str(r.get("outcome_draw", "") or "-")[:12],
            str(r["outcome_b"])[:22],
            f'{float(r["price_a_cents"]):.2f}c',
            f'{float(r["price_draw_cents"]):.2f}c' if pd.notna(r.get("price_draw_cents")) else "-",
            f'{float(r["price_b_cents"]):.2f}c',
            f'{float(r["liquidity"]):,.0f}',
            _fmt_dt(r.get("start_date")),
        )
    console.print(t)


def render_ev_table(df: pd.DataFrame, sport: str, max_rows: int = 25):
    t = Table(title=f"Model EV Overlay ({sport.upper()})", box=box.SIMPLE_HEAVY, show_lines=False)
    cols = [
        ("#", "right"),
        ("Matchup", "left"),
        ("P(A)", "right"),
        ("P(Draw)", "right"),
        ("P(B)", "right"),
        ("A Px", "right"),
        ("D Px", "right"),
        ("B Px", "right"),
        ("Edge A", "right"),
        ("Edge D", "right"),
        ("Edge B", "right"),
        ("Best", "left"),
        ("Action", "left"),
    ]
    for c, j in cols:
        t.add_column(c, justify=j, overflow="fold")

    for i, (_, r) in enumerate(df.head(max_rows).iterrows(), 1):
        side = str(r.get("best_side", "")).strip()
        action = str(r.get("action", "")).strip()
        status = str(r.get("status", "")).strip()
        a_prob = r.get("prob_a_cal", r.get("prob_home_cal", float("nan")))
        d_prob = r.get("prob_draw_cal", float("nan"))
        b_prob = r.get("prob_b_cal", r.get("prob_away_cal", float("nan")))
        edge_a = r.get("edge_a_cents", r.get("edge_home_cents", float("nan")))
        edge_d = r.get("edge_draw_cents", float("nan"))
        edge_b = r.get("edge_b_cents", r.get("edge_away_cents", float("nan")))
        team_a = r.get("fighter_a_resolved", r.get("team_a", r.get("home_team", r.get("orig_outcome_a", ""))))
        team_b = r.get("fighter_b_resolved", r.get("team_b", r.get("away_team", r.get("orig_outcome_b", ""))))
        if team_a in ("nan", "None") or team_b in ("nan", "None"):
            oa = str(r.get("orig_outcome_a", r.get("outcome_a", r.get("team_a_input", ""))))
            ob = str(r.get("orig_outcome_b", r.get("outcome_b", r.get("team_b_input", ""))))
            matchup = f"{oa} vs {ob}".strip()
        else:
            matchup = f"{team_a} vs {team_b}"

        t.add_row(
            str(i),
            matchup,
            f"{float(a_prob):.3f}" if pd.notna(a_prob) else "-",
            f"{float(d_prob):.3f}" if pd.notna(d_prob) else "-",
            f"{float(b_prob):.3f}" if pd.notna(b_prob) else "-",
            f'{float(r.get("price_a_cents", r.get("price_home_cents", float("nan")))):.2f}c' if pd.notna(r.get("price_a_cents", r.get("price_home_cents", float("nan")))) else "-",
            f'{float(r.get("price_draw_cents", float("nan"))):.2f}c' if pd.notna(r.get("price_draw_cents", float("nan"))) else "-",
            f'{float(r.get("price_b_cents", r.get("price_away_cents", float("nan")))):.2f}c' if pd.notna(r.get("price_b_cents", r.get("price_away_cents", float("nan")))) else "-",
            f"{float(edge_a):+.2f}" if pd.notna(edge_a) else "-",
            f"{float(edge_d):+.2f}" if pd.notna(edge_d) else "-",
            f"{float(edge_b):+.2f}" if pd.notna(edge_b) else "-",
            side,
            f"[green]{action}[/green]"
            if action == "BUY"
            else (f"[yellow]{action}[/yellow]" if action else (status if status else "-")),
        )
    console.print(t)


def build_parser():
    p = argparse.ArgumentParser(description="Polymarket Sports CLI with model EV overlay")
    p.add_argument("--sport", choices=sorted(SPORT_TAG_SLUGS.keys()), help="Sport to fetch")
    p.add_argument("--pages", type=int, default=6, help="Number of API pages to fetch")
    p.add_argument("--page-size", type=int, default=100, help="Rows per page")
    p.add_argument("--query", default="", help="Title filter (contains)")
    p.add_argument("--min-liquidity", type=float, default=0.0, help="Minimum event liquidity")
    p.add_argument("--predict", action="store_true", help="Run model prediction overlay")
    p.add_argument("--out-csv", default="", help="Optional output CSV path")
    p.add_argument("--interactive", action="store_true", help="Interactive dashboard mode")
    p.add_argument("--sports-compare", action="store_true", help="Build all-sports comparison table")
    p.add_argument("--paper", action="store_true", help="Paper trade BUY recommendations into virtual ledger")
    p.add_argument("--paper-stake-usd", type=float, default=25.0, help="Paper stake per BUY position")
    p.add_argument("--paper-summary", action="store_true", help="Show current paper portfolio summary")
    p.add_argument(
        "--soccer-include-non-europe",
        action="store_true",
        help="For soccer, include non-Europe/non-club markets (default is Europe club-only)",
    )
    return p


def fetch_market_frame(
    client: GammaClient,
    sport: str,
    pages: int,
    page_size: int,
    query: str,
    min_liquidity: float,
    soccer_club_only: bool = True,
) -> pd.DataFrame:
    rows = client.fetch_game_markets(
        sport=sport,
        pages=pages,
        page_size=page_size,
        query=query or None,
        min_liquidity=min_liquidity,
        soccer_club_only=soccer_club_only,
    )
    df = _build_market_df(rows)
    if df.empty:
        return df
    df = df.sort_values(["start_date", "liquidity"], ascending=[True, False]).reset_index(drop=True)
    return df


def run_once(args):
    client = GammaClient()
    df = fetch_market_frame(
        client,
        args.sport,
        args.pages,
        args.page_size,
        args.query,
        args.min_liquidity,
        soccer_club_only=not args.soccer_include_non_europe,
    )
    if df.empty:
        console.print(Panel.fit(f"No active two-outcome matchup markets found for sport='{args.sport}'.", border_style="yellow"))
        return

    render_market_table(df)

    out_df = df.copy()
    pred = pd.DataFrame()
    if args.predict:
        try:
            pred = predict_with_model(args.sport, df)
            if pred is not None and not pred.empty:
                render_ev_table(pred, args.sport)
                out_df = pd.concat([df.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
            else:
                console.print(Panel.fit("Model returned no rows.", border_style="yellow"))
        except Exception as e:
            console.print(Panel.fit(f"Model overlay unavailable: {e}", border_style="red"))

    if args.paper:
        if pred is None or pred.empty:
            console.print(Panel.fit("Paper mode requires model predictions. Use --predict.", border_style="yellow"))
        else:
            merged = pd.concat([df.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
            new_pos = build_new_positions_from_reco(merged, stake_usd=args.paper_stake_usd, only_buy=True)
            append_positions(new_pos)
            console.print(
                Panel.fit(
                    f"Paper trades placed: {len(new_pos)}\nStake per trade: ${args.paper_stake_usd:.2f}\nLedger: PolymarketCLI/paper/paper_ledger.csv",
                    border_style="green",
                )
            )

    if args.out_csv:
        out_path = Path(args.out_csv).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        console.print(f"[green]Saved:[/green] {out_path}")


def interactive_shell():
    console.print(Panel.fit("Polymarket Sports Terminal\nDeep market browse + model EV overlay", border_style="cyan"))
    client = GammaClient()

    while True:
        sport = Prompt.ask("Sport", choices=sorted(SPORT_TAG_SLUGS.keys()), default="nba")
        pages = int(Prompt.ask("Pages", default="6"))
        page_size = int(Prompt.ask("Page size", default="100"))
        query = Prompt.ask("Query filter (optional)", default="")
        min_liq = float(Prompt.ask("Min liquidity", default="0"))
        with_model = Confirm.ask("Run model overlay?", default=True)
        do_paper = Confirm.ask("Paper trade BUY signals?", default=False) if with_model else False

        df = fetch_market_frame(client, sport, pages, page_size, query, min_liq, soccer_club_only=True)
        if df.empty:
            console.print(Panel.fit("No markets found with those filters.", border_style="yellow"))
        else:
            render_market_table(df)
            if with_model:
                try:
                    pred = predict_with_model(sport, df)
                    if pred is not None and not pred.empty:
                        render_ev_table(pred, sport)
                        if do_paper:
                            merged = pd.concat([df.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
                            new_pos = build_new_positions_from_reco(merged, stake_usd=25.0, only_buy=True)
                            append_positions(new_pos)
                            console.print(Panel.fit(f"Paper trades placed: {len(new_pos)}", border_style="green"))
                        if Confirm.ask("Export merged CSV?", default=False):
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                            out = Path("PolymarketCLI") / "outputs" / f"polymarket_{sport}_{ts}.csv"
                            out.parent.mkdir(parents=True, exist_ok=True)
                            pd.concat([df.reset_index(drop=True), pred.reset_index(drop=True)], axis=1).to_csv(out, index=False)
                            console.print(f"[green]Saved:[/green] {out.resolve()}")
                except Exception as e:
                    console.print(Panel.fit(f"Model overlay failed: {e}", border_style="red"))

        if not Confirm.ask("Fetch another slate?", default=True):
            break


def main_from_namespace(args):
    if args.interactive:
        interactive_shell()
        return

    if args.paper_summary:
        led = load_ledger()
        if led.empty:
            console.print(Panel.fit("Paper ledger is empty.", border_style="yellow"))
            return
        # Mark-to-market using latest markets across sports
        client = GammaClient()
        latest_all = []
        for s in sorted(SPORT_TAG_SLUGS.keys()):
            d = fetch_market_frame(client, s, pages=4, page_size=100, query="", min_liquidity=0.0, soccer_club_only=True)
            if not d.empty:
                latest_all.append(d)
        latest = pd.concat(latest_all, ignore_index=True) if latest_all else pd.DataFrame()
        mtm = mark_to_market(led, latest)
        summary = summarize_portfolio(mtm)
        t = Table(title="Paper Portfolio Summary", box=box.SIMPLE_HEAVY)
        t.add_column("Metric")
        t.add_column("Value", justify="right")
        t.add_row("Open Positions", str(summary["positions_open"]))
        t.add_row("Total Stake", f'${summary["total_stake_usd"]:.2f}')
        t.add_row("Market Value", f'${summary["market_value_usd"]:.2f}')
        t.add_row("Unrealized PnL", f'${summary["unrealized_pnl_usd"]:.2f}')
        console.print(t)
        out = Path("PolymarketCLI") / "paper" / f'paper_mtm_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        out.parent.mkdir(parents=True, exist_ok=True)
        mtm.to_csv(out, index=False)
        console.print(f"[green]Saved:[/green] {out.resolve()}")
        return

    if args.sports_compare:
        client = GammaClient()
        sports = sorted(SPORT_TAG_SLUGS.keys())
        rows: List[Dict[str, object]] = []
        for s in sports:
            df = fetch_market_frame(
                client,
                s,
                pages=args.pages,
                page_size=args.page_size,
                query=args.query,
                min_liquidity=args.min_liquidity,
                soccer_club_only=not args.soccer_include_non_europe,
            )
            rec = {
                "sport": s,
                "markets_found": int(len(df)),
                "avg_liquidity": float(df["liquidity"].mean()) if not df.empty else 0.0,
                "avg_price_sum_cents": float((df["price_a_cents"] + df["price_b_cents"]).mean()) if not df.empty else 0.0,
                "model_supported": s in {"nba", "nfl", "ufc", "soccer"},
                "model_rows": 0,
                "buy_signals": 0,
                "avg_best_edge_cents": None,
                "max_best_edge_cents": None,
            }
            if rec["model_supported"] and not df.empty:
                try:
                    pred = predict_with_model(s, df)
                    if pred is not None and not pred.empty:
                        rec["model_rows"] = int(len(pred))
                        buys = pred[pred["action"].astype(str).str.upper() == "BUY"] if "action" in pred.columns else pred.iloc[0:0]
                        rec["buy_signals"] = int(len(buys))
                        if "best_edge_cents" in pred.columns:
                            rec["avg_best_edge_cents"] = float(pd.to_numeric(pred["best_edge_cents"], errors="coerce").mean())
                            rec["max_best_edge_cents"] = float(pd.to_numeric(pred["best_edge_cents"], errors="coerce").max())
                except Exception:
                    pass
            rows.append(rec)

        comp = pd.DataFrame(rows).sort_values("sport").reset_index(drop=True)
        t = Table(title="All Sports Comparison", box=box.SIMPLE_HEAVY)
        for c in ["sport", "markets_found", "model_supported", "model_rows", "buy_signals", "avg_best_edge_cents", "max_best_edge_cents", "avg_liquidity"]:
            t.add_column(c, justify="right" if c != "sport" else "left")
        for _, r in comp.iterrows():
            t.add_row(
                str(r["sport"]).upper(),
                str(int(r["markets_found"])),
                "yes" if bool(r["model_supported"]) else "no",
                str(int(r["model_rows"])),
                str(int(r["buy_signals"])),
                f'{float(r["avg_best_edge_cents"]):.2f}' if pd.notna(r["avg_best_edge_cents"]) else "-",
                f'{float(r["max_best_edge_cents"]):.2f}' if pd.notna(r["max_best_edge_cents"]) else "-",
                f'{float(r["avg_liquidity"]):,.0f}',
            )
        console.print(t)
        out = Path(args.out_csv).expanduser().resolve() if args.out_csv else (Path("PolymarketCLI") / "outputs" / f"sports_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv").resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        comp.to_csv(out, index=False)
        console.print(f"[green]Saved:[/green] {out}")
        return

    if not args.sport:
        raise ValueError("Either pass --interactive or provide --sport.")
    run_once(args)


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        main_from_namespace(args)
    except ValueError as e:
        parser.error(str(e))


if __name__ == "__main__":
    main()
