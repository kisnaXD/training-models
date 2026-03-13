# NBA Scraper (NBA-only)

This folder contains a **mainstream NBA-only** historical data scraper.

## Files
- `nba_scraper.py`: pulls NBA teams, franchise history, team game logs, and merged game-level rows.
- `requirements_nba.txt`: package list for the scraper.

## Install
```powershell
python -m pip install -r nba\requirements_nba.txt
```

## Run
```powershell
python nba\nba_scraper.py --start-season 2000 --end-season 2025 --outdir nba_data
```

## Outputs
Inside `--outdir` (default `nba_data`):
- `teams.csv`
- `franchise_history.csv`
- `raw_team_game_logs.csv`
- `games.csv`
- `metadata.json`

## Notes
- League is hard-coded to NBA (`league_id = 00`), so this excludes WNBA and college.
- Game logs include both Regular Season and Playoffs.
- Scraper uses retries + backoff to handle transient `stats.nba.com` failures.

## Train Pipeline
```powershell
python nba\nba_model_pipeline.py --data-dir nba_data --outdir nba_outputs
```

Outputs in `nba_outputs`:
- `nba_feature_table.csv`
- `nba_model_report.csv`
- `nba_xgb_model.pkl`
- `nba_imputer.pkl`
- `nba_isotonic.pkl`
- `nba_artifacts_meta.json`
