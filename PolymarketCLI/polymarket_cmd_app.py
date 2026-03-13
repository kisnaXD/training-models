#!/usr/bin/env python3
"""Full-screen style CMD app launcher for Polymarket sports terminal."""

from __future__ import annotations

import os
from argparse import Namespace
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

import polymarket_sports_cli as cli

console = Console()


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def header():
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    console.print(Panel.fit(f"Polymarket Sports Command Center\n{t}", border_style="cyan"))


def menu():
    t = Table(show_header=True, header_style="bold cyan")
    t.add_column("Option", justify="right")
    t.add_column("Action")
    t.add_row("1", "Browse Markets + Model EV")
    t.add_row("2", "Sports Comparison Table")
    t.add_row("3", "Paper Portfolio Summary")
    t.add_row("4", "Quick Soccer (Europe Club-Only) EV")
    t.add_row("5", "Exit")
    console.print(t)


def run_browse():
    sport = Prompt.ask("Sport", choices=["nba", "nfl", "nhl", "ufc", "soccer"], default="nba")
    pages = int(Prompt.ask("Pages", default="6"))
    page_size = int(Prompt.ask("Page Size", default="100"))
    query = Prompt.ask("Query Filter (blank for none)", default="")
    min_liq = float(Prompt.ask("Min Liquidity", default="0"))
    predict = Prompt.ask("Run model? (y/n)", choices=["y", "n"], default="y") == "y"
    paper = Prompt.ask("Paper trade BUY signals? (y/n)", choices=["y", "n"], default="n") == "y"
    stake = float(Prompt.ask("Paper stake USD", default="25")) if paper else 25.0
    out_csv = Prompt.ask("Export CSV path (blank skip)", default="")

    args = Namespace(
        sport=sport,
        pages=pages,
        page_size=page_size,
        query=query,
        min_liquidity=min_liq,
        predict=predict,
        out_csv=out_csv,
        interactive=False,
        sports_compare=False,
        paper=paper,
        paper_stake_usd=stake,
        paper_summary=False,
        soccer_include_non_europe=False,
    )
    cli.run_once(args)


def run_compare():
    out_csv = Prompt.ask("Export CSV path", default=f"PolymarketCLI/outputs/sports_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    args = Namespace(
        sport=None,
        pages=4,
        page_size=100,
        query="",
        min_liquidity=0.0,
        predict=False,
        out_csv=out_csv,
        interactive=False,
        sports_compare=True,
        paper=False,
        paper_stake_usd=25.0,
        paper_summary=False,
        soccer_include_non_europe=False,
    )
    cli.main_from_namespace(args)


def run_paper_summary():
    args = Namespace(
        sport=None,
        pages=4,
        page_size=100,
        query="",
        min_liquidity=0.0,
        predict=False,
        out_csv="",
        interactive=False,
        sports_compare=False,
        paper=False,
        paper_stake_usd=25.0,
        paper_summary=True,
        soccer_include_non_europe=False,
    )
    cli.main_from_namespace(args)


def run_soccer_quick():
    args = Namespace(
        sport="soccer",
        pages=8,
        page_size=100,
        query="",
        min_liquidity=0.0,
        predict=True,
        out_csv=f"PolymarketCLI/outputs/soccer_ev_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        interactive=False,
        sports_compare=False,
        paper=False,
        paper_stake_usd=25.0,
        paper_summary=False,
        soccer_include_non_europe=False,
    )
    cli.run_once(args)


def main():
    while True:
        clear()
        header()
        menu()
        choice = Prompt.ask("Choose", choices=["1", "2", "3", "4", "5"], default="1")
        clear()
        header()
        try:
            if choice == "1":
                run_browse()
            elif choice == "2":
                run_compare()
            elif choice == "3":
                run_paper_summary()
            elif choice == "4":
                run_soccer_quick()
            else:
                console.print("Goodbye.")
                break
        except Exception as e:
            console.print(Panel.fit(f"Error: {e}", border_style="red"))
        Prompt.ask("\nPress Enter to return to menu", default="")


if __name__ == "__main__":
    main()

