# Polymarket Sports CLI

Rich terminal dashboard to:
- Deep-search Polymarket sports markets via Gamma API (pagination + filtering)
- Show live market prices and liquidity
- Overlay your local model EV recommendations (NBA / NFL / UFC)
- Overlay your local model EV recommendations (NBA / NFL / UFC / Soccer)
- Paper-trade BUY signals into a virtual ledger
- Compare all sports in one table

## Run

```bash
python PolymarketCLI/polymarket_sports_cli.py --interactive
```

Full CMD interface launcher:

```bash
python polymarket_terminal.py
```

or one-shot:

```bash
python PolymarketCLI/polymarket_sports_cli.py --sport nba --pages 8 --page-size 100 --predict --out-csv PolymarketCLI/outputs/nba_ev.csv
```

Soccer (Europe club-only is default):

```bash
python PolymarketCLI/polymarket_sports_cli.py --sport soccer --pages 8 --page-size 100 --predict
```

Include non-Europe/non-club soccer markets:

```bash
python PolymarketCLI/polymarket_sports_cli.py --sport soccer --pages 8 --page-size 100 --soccer-include-non-europe
```

Paper mode:

```bash
python PolymarketCLI/polymarket_sports_cli.py --sport ufc --pages 6 --page-size 100 --predict --paper --paper-stake-usd 20 --out-csv PolymarketCLI/outputs/ufc_paper.csv
python PolymarketCLI/polymarket_sports_cli.py --paper-summary
```

All-sports comparison:

```bash
python PolymarketCLI/polymarket_sports_cli.py --sports-compare --pages 4 --page-size 100 --out-csv PolymarketCLI/outputs/sports_compare_latest.csv
```

## Notes

- Supported model overlays: `nba`, `nfl`, `ufc`, `soccer`
- `nhl` market browsing works, but model overlay is not wired yet in this CLI.
- This tool currently focuses on discovery + recommendation. It does not auto-place orders.
- Soccer overlay uses a 3-way model (`team A / draw / team B`) and applies no-draw conditioning when market is only two-sided.
