# Scripts

This directory contains the data download and analysis utilities used in the workflow.

## download_gofer_zenodo.py

Download GOFER product files from Zenodo.

- Inputs: Zenodo record id or DOI (optional).
- Outputs: files saved under a target directory (default: data/zenodo).
- Example:
  - python3 scripts/download_gofer_zenodo.py --target data/zenodo

## ee_download_confidence_stack.py

Download GOFER confidence stacks (GOES East/West MaxConf) from Earth Engine as multi-band GeoTIFFs. Handles band chunking and CRS override.

- Inputs: fire name/year/source OR explicit asset id.
- Outputs: GeoTIFF (single or multiple _partXX.tif files).
- Example:
  - python3 scripts/ee_download_confidence_stack.py --fire-name "Kincade" --year 2019 --source east --output data/ee_exports/Kincade_2019_GOESEast_MaxConf.tif

## gofer_confidence_to_json.py

Convert GOFER confidence GeoTIFF stacks to a JSON time series with metadata.

- Inputs: one or more GeoTIFFs (chronological order).
- Outputs: JSON with { metadata, data }.
- Example:
  - python3 scripts/gofer_confidence_to_json.py --input data/ee_exports/Kincade_2019_GOESEast_MaxConf.tif --output data/ee_exports/Kincade_2019_GOESEast_MaxConf.json --fire-name "Kincade" --year 2019 --source "GOES-East" --start-time "2019-10-24T04:00:00Z" --pretty

## ee_download_rtma.py

Download RTMA hourly grids aligned to a GOES GeoTIFF and write a manifest.

- Inputs: GOES GeoTIFF (for bounds/CRS), start and end time.
- Outputs: rtma_<VAR>_partXX.tif files and rtma_manifest.json.
- Example:
  - python3 scripts/ee_download_rtma.py --goes-tif data/ee_exports/August_Complex_2020_GOESEast_MaxConf_part01.tif --start 2020-08-16T21:00:00Z --end 2020-10-18T21:00:00Z --output-dir data/rtma/august --chunk-hours 24

## rtma_to_json.py

Convert RTMA GeoTIFF chunks into a single JSON with optional normalization.

- Inputs: rtma_manifest.json.
- Outputs: JSON with per-variable time series grids.
- Example:
  - python3 scripts/rtma_to_json.py --manifest data/rtma/august/rtma_manifest.json --output data/rtma/august/rtma_normalized.json --normalize zscore

## firms_viirs_to_json.py

Download FIRMS VIIRS data for a bbox and date range, then grid into GOFER-style JSON.

- Inputs: FIRMS MAP_KEY, bbox, start date, day range, source.
- Outputs: JSON (optionally raw CSV).
- Example:
  - FIRMS_MAP_KEY=YOUR_KEY python3 scripts/firms_viirs_to_json.py --fire-name "Kincade" --year 2019 --bbox -122.96 38.50 -122.59 38.87 --start-date 2019-10-24 --day-range 7 --source VIIRS_SNPP_SP --output data/viirs/Kincade_2019_VIIRS_SNPP_SP.json

## synoptic_raws_fetch.py

Fetch RAWS stations and hourly data from the Synoptic Data API.

- Inputs: SYNOPTIC token, bbox, start/end (YYYYmmddHHMM).
- Outputs: raws_stations.json and raws_timeseries_hourly.json.
- Example:
  - SYNOPTIC_TOKEN=YOUR_TOKEN python3 scripts/synoptic_raws_fetch.py --bbox -122.96 38.50 -122.59 38.87 --start 201910240000 --end 201910302300 --output data/raws/kincade

## raws_normalize.py

Normalize RAWS hourly time series per station.

- Inputs: raws_timeseries_hourly.json.
- Outputs: normalized JSON with per-station stats.
- Example:
  - python3 scripts/raws_normalize.py --input data/raws/kincade/raws_timeseries_hourly.json --output data/raws/kincade/raws_timeseries_hourly_normalized.json --method zscore

## raws_nearest_station_grid.py

Create nearest-RAWS-station index and distance grids aligned to a GOES GeoTIFF.

- Inputs: GOES GeoTIFF and raws_stations.json.
- Outputs: raws_station_index.npy, raws_station_distance_km.npy, raws_station_lookup.json.
- Example:
  - python3 scripts/raws_nearest_station_grid.py --goes-tif data/ee_exports/August_Complex_2020_GOESEast_MaxConf_part01.tif --stations-json data/raws/august/raws_stations.json --output-dir data/raws/august/grid

## run_locational_regressions.py

Run spread and continuation logistic regressions for a single fire.

- Inputs: GOES JSON and RTMA manifest.
- Outputs: regression_report.json, regression_report.txt, probability_maps_spread.json, probability_maps_continue.json.
- Example:
  - python3 scripts/run_locational_regressions.py --goes-json data/ee_exports/August_Complex_2020_GOESEast_MaxConf.json --rtma-manifest data/rtma/august_24h/rtma_manifest.json --threshold 0.1 --output-dir data/analysis/august_locational --neg-ratio 5 --max-samples 200000 --prob-hours 3

## Multi-fire logistic analysis

Multi-fire logistic runs are handled in `docs/neighbor_cell_confidence_regression.ipynb`
under the **Multi-Fire Logistic Runs** section.

## build_multires_manifest.py

Build a versioned multi-resolution manifest (`manifest_version=2.0`) from a config JSON.

- Inputs: multires config JSON (see `docs/examples/multires_config.example.json`).
- Outputs: normalized manifest with flattened variable catalog.
- Example:
  - python3 scripts/build_multires_manifest.py --config docs/examples/multires_config.example.json --output data/multi_fire/North_Complex/multires_manifest_v2.json

## validate_multires_manifest.py

Validate a multi-resolution manifest (required keys, source uniqueness, variable linkage, basic type/range checks).

- Inputs: multires manifest JSON.
- Outputs: validation summary to stdout (non-zero exit on failure).
- Example:
  - python3 scripts/validate_multires_manifest.py --manifest data/multi_fire/North_Complex/multires_manifest_v2.json
