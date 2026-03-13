# Soccer (European Club-Only)

This pipeline is intentionally **club-focused** to avoid player-identity leakage across
national teams and multiple club contexts.

## Scope
- European domestic club leagues:
  - Premier League, La Liga, Serie A, Bundesliga, Ligue 1
  - Eredivisie, Primeira Liga, Belgian Pro League, Scottish Premiership, Turkish Super Lig
- UEFA club tournaments:
  - Champions League, Europa League, Conference League

## Install
```powershell
python -m pip install -r Soccer\requirements_soccer.txt
```

## 1) Scrape
```powershell
python Soccer\soccer_scraper.py --start-season 2010 --end-season 2025 --outdir Soccer\soccer_data
```

## 2) Train
```powershell
python Soccer\soccer_model_pipeline.py --data-dir Soccer\soccer_data --outdir Soccer\soccer_outputs
```

## Outputs
`Soccer\soccer_outputs`:
- `soccer_feature_table.csv`
- `soccer_model_report.csv`
- `soccer_xgb_model.pkl`
- `soccer_imputer.pkl`
- `soccer_isotonic.pkl`
- `soccer_artifacts_meta.json`
## 3-way model (Home / Draw / Away)

Train:

```bash
python Soccer/soccer_model_pipeline_3way.py --data-dir Soccer/soccer_data --outdir Soccer/soccer_outputs_3way --min-mirrored-rows 122000
```

EV CLI:

```bash
python Soccer/soccer_ev_cli_3way.py --team-a "Real Madrid CF" --team-b "Manchester City FC" --price-a-cents 55 --price-b-cents 45 --competition-code CL --competition-name "UEFA Champions League"
```
