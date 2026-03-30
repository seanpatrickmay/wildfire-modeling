# Repository Documentation

This repository contains the GOFER workflow (GOES Observed Fire Event Representation) plus helper scripts for downloading data, converting it to JSON, and running regression analyses.

## Quickstart (local workflows)

1) Authenticate Earth Engine and set a project (if needed).
2) Download GOES confidence stacks for a fire.
3) Convert the GeoTIFF stacks to JSON.
4) Download RTMA hourly meteorology aligned to the GOES grid.
5) Run regressions (single fire) and notebook-driven multi-fire logistic analysis.

## Key data sources

- GOFER GOES confidence stacks (GOESEast_MaxConf, GOESWest_MaxConf) in Earth Engine.
- NOAA/NWS RTMA hourly meteorology in Earth Engine.
- Optional: FIRMS VIIRS point detections (FIRMS API).
- Optional: RAWS (Synoptic Data API).

## Main pipelines

### A) GOES confidence -> JSON
- Download GeoTIFF stacks from Earth Engine.
- Convert to GOFER-style JSON for downstream modeling and visualization.

### B) RTMA -> JSON
- Download hourly RTMA grids aligned to the GOES region.
- Build a manifest, then optionally normalize into a single JSON.

### C) Regression analysis
- Single-fire regressions: spread + continuation models.
- Multi-fire logistic runs: use `docs/neighbor_cell_confidence_regression.ipynb`.

### D) Optional data
- FIRMS VIIRS: grid point detections into a GOFER-style JSON.
- RAWS: download station time series, normalize, and map nearest station to GOES grid.

### E) Higher-resolution planning artifacts
- `higher_resolution_data_options.md`: source comparison matrix and recommended CONUS upgrade path.
- `multires_data_contract.md`: versioned manifest v2 schema for multi-resolution covariates.
- `examples/multires_config.example.json`: starter config for manifest generation.

## Where to look next

- scripts/README.md: command-line tools and outputs.
- data/README.md: generated data layout and file formats.
- gee/README.md: Earth Engine scripts and assets.
- R/README.md: R post-processing scripts.
- webapp/README.md: local JSON visualization app.
- docs/neighbor_cell_confidence_regression.ipynb: neighbor-cell logistic notebook with single-fire and multi-fire blocks.
- docs/higher_resolution_data_options.md: data-source research and recommendation.
- docs/multires_data_contract.md: multiresolution manifest contract.
