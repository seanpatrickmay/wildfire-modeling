# Wildfire Prediction Data + Regression Workflow

This repository is a local data-ingestion and modeling workflow used to:

- download wildfire-related datasets (GOES/GOFER confidence stacks and aligned meteorology)
- normalize local credentials for optional APIs via environment variables or `.env`
- convert gridded data into GOFER-style JSON time series
- run locational logistic regressions (spread + continuation)
- summarize results in regression reports and Jupyter notebooks

If you are looking for the GOFER product generation pipeline itself, it lives under `gee/` (Earth Engine) and `R/` (post-processing). The day-to-day work in this repo is the data download + regression analysis pipeline.

## Pipeline at a Glance

1. Download GOES/GOFER confidence stacks from Earth Engine.
2. Convert GOES stacks into GOFER-style JSON.
3. Download RTMA hourly meteorology aligned to the GOES grid.
4. Normalize RTMA into a single JSON (optional but commonly used).
5. Run locational logistic regressions (single fire or aggregated).
6. Review regression reports and notebooks.

## Data Sources

- GOES/GOFER confidence stacks (Earth Engine assets).
- NOAA/NWS RTMA hourly meteorology (Earth Engine).
- Optional FIRMS VIIRS detections (NASA FIRMS API).
- Optional RAWS station data (Synoptic Data API).

## Environment and Credentials

Earth Engine downloads require `earthengine-api` and authentication. Optional APIs (FIRMS, Synoptic) use a simple credential normalization pattern: scripts check CLI flags first, then environment variables, then a local `.env` file, and accept multiple common key names.

- FIRMS tokens: `FIRMS_MAP_KEY` (also accepts `MAP_KEY`, `map-key`, `map_key`).
- Synoptic tokens: `SYNOPTIC_TOKEN` (also accepts `SYNOPTIC_API_TOKEN`, `SYNOPTIC_PUBLIC_TOKEN`, `TOKEN`, `token`).

Example `.env`:

```text
FIRMS_MAP_KEY=your_firms_key
SYNOPTIC_TOKEN=your_synoptic_token
```

## Dependencies

- Python 3.10+
- `earthengine-api` for Earth Engine downloads
- `numpy`, `rasterio`, `scikit-learn` for conversion and regression

There is no lockfile in this repo, so install dependencies in your environment of choice.

## Quickstart (Single Fire)

1. Download GOES confidence stack (Earth Engine):

```bash
python3 scripts/ee_download_confidence_stack.py \
  --fire-name "Kincade" --year 2019 --source east \
  --output data/ee_exports/Kincade_2019_GOESEast_MaxConf.tif
```

2. Convert GOES stack to JSON:

```bash
python3 scripts/gofer_confidence_to_json.py \
  --input data/ee_exports/Kincade_2019_GOESEast_MaxConf.tif \
  --output data/ee_exports/Kincade_2019_GOESEast_MaxConf.json \
  --fire-name "Kincade" --year 2019 --source "GOES-East" \
  --start-time "2019-10-24T04:00:00Z"
```

3. Download RTMA aligned to GOES geometry:

```bash
python3 scripts/ee_download_rtma.py \
  --goes-tif data/ee_exports/Kincade_2019_GOESEast_MaxConf.tif \
  --start 2019-10-24T04:00:00Z --end 2019-10-31T04:00:00Z \
  --output-dir data/rtma/kincade --chunk-hours 24
```

4. Normalize RTMA into a single JSON (optional):

```bash
python3 scripts/rtma_to_json.py \
  --manifest data/rtma/kincade/rtma_manifest.json \
  --output data/rtma/kincade/rtma_normalized.json \
  --normalize zscore
```

5. Run locational regressions:

```bash
python3 scripts/run_locational_regressions.py \
  --goes-json data/ee_exports/Kincade_2019_GOESEast_MaxConf.json \
  --rtma-manifest data/rtma/kincade/rtma_manifest.json \
  --threshold 0.1 --neg-ratio 5 --max-samples 200000 \
  --output-dir data/analysis/kincade
```

## Aggregate (Multi-Fire) Workflow

Run the end-to-end multi-fire pipeline (download, convert, normalize, regress):

```bash
python3 scripts/run_multi_fire_pipeline.py \
  --output-dir data/multi_fire --chunk-size 256 --chunk-hours 24
```

Or build your own config and run the aggregate regression directly:

```bash
python3 scripts/run_locational_regressions_aggregate.py \
  --config data/multi_fire/aggregate_config.json \
  --threshold 0.1 --neg-ratio 0 --max-samples-per-fire 0 \
  --output data/multi_fire/aggregate_regression_report.json
```

## Outputs

- `data/analysis/<fire>/regression_report.json`
- `data/analysis/<fire>/regression_report.txt`
- `data/analysis/<fire>/probability_maps_spread.json`
- `data/analysis/<fire>/probability_maps_continue.json`
- `data/multi_fire/aggregate_regression_report.json`

See `data/README.md` for a fuller output layout.

## Notebooks and Visualization

- `docs/aggregate_regression_report.ipynb` summarizes aggregated regression results.
- `docs/higher_resolution_data_options.md` documents higher-resolution data candidates and recommendations.
- `docs/multires_data_contract.md` defines a versioned manifest for multi-resolution covariates.
- `webapp/` provides a lightweight viewer for GOFER-style JSON time series.

## Repository Layout

- `scripts/` CLI utilities for downloads, conversions, and regressions
- `data/` generated outputs and intermediate artifacts
- `docs/` documentation and regression report notebooks
- `gee/` Earth Engine pipeline for GOFER product generation
- `R/` R post-processing scripts for GOFER
- `webapp/` local JSON visualization

## Notes and Scope

- This is a research workflow, not a production forecasting system.
- GOFER product assets are external; use `scripts/download_gofer_zenodo.py` if you want the published GOFER files without Earth Engine.
- For more details on scripts and data formats, see `docs/README.md` and `scripts/README.md`.
