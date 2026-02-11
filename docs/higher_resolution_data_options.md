# Higher-Resolution Data Options for Wildfire Modeling (CONUS)

## Purpose
This document evaluates higher-resolution data sources to improve the current workflow baseline:

- GOES/GOFER confidence at ~2 km (`scripts/ee_download_confidence_stack.py` default scale `2004.017315487541`)
- RTMA meteorology at 2.5 km (`scripts/ee_download_rtma.py` scale `2500`)

Target priorities for this upgrade path:

1. Spatial detail first.
2. CONUS coverage.
3. Historical/research use (not strict low-latency operations).
4. Open datasets first, with paid options as optional fallback.

## Executive Recommendation
Adopt a multi-resolution stack with:

1. **Model grid at 375 m** using VIIRS-based fire observations for dynamic fire signal.
2. **Static context at 10-30 m** (3DEP DEM + LANDFIRE + NLCD + HLS/Sentinel/Landsat-derived indices), aggregated to model grid.
3. **Meteorology at 800 m-1 km daily** (PRISM and/or Daymet) plus optional MRMS 1 km precipitation where sub-daily precip structure is needed.
4. Continue GOES + RTMA as coarse baseline features for continuity and ablation comparisons.

## Data Source Matrix

| Source | Native Resolution | Temporal Cadence | Coverage | Access | License/Cost | Strengths | Limitations |
|---|---:|---|---|---|---|---|---|
| FIRMS VIIRS active fire | 375 m (point detections) | Near-real-time + historical | Global (CONUS available) | API/CSV | Open (token required) | Major spatial improvement over GOES confidence for fire localization | Not a complete radiance field; detection-driven |
| HLS (Landsat + Sentinel harmonized) | 30 m | ~2-3 day effective revisit (cloud-limited) | Global land incl. CONUS | NASA Earthdata | Open | High-quality retrospective surface context at high resolution | Cloud contamination; not hourly |
| Sentinel-2 L2A | 10/20/60 m | ~5 day revisit (constellation), cloud-limited | Global | Copernicus hubs/cloud platforms | Open | Very high spatial detail for vegetation/burn-related indices | Temporal gaps from clouds |
| Landsat Collection 2 | 30 m | ~8-16 day revisit (platform mix), cloud-limited | Global | USGS/EE/cloud platforms | Open | Long historical continuity, robust products | Sparse revisit for fast fire dynamics |
| PRISM climate | ~800 m (4 km also available) | Daily/monthly | CONUS | PRISM downloads/services | Open for research use | Better spatial meteorology detail than 2.5 km RTMA for historical modeling | Not hourly NRT meteorology |
| Daymet | 1 km | Daily | North America | ORNL DAAC | Open | Long, consistent daily met climate forcing | Daily only |
| MRMS | 1 km | Minutes to hourly aggregations | CONUS | NOAA/NCEI/NODD pathways | Open | Fine-resolution precip structure | Product handling complexity |
| 3DEP DEM | 10 m / 30 m | Static | CONUS | USGS | Open | High-quality terrain features (slope/aspect/topographic position) | Static only |
| LANDFIRE | 30 m | Periodic release | CONUS | LANDFIRE | Open | Fuels/vegetation structure directly relevant to spread potential | Update cycles slower than weather/fire |
| NLCD | 30 m | Multi-year epochs | CONUS | USGS | Open | Standard land cover covariates | Not event-dynamic |
| Planet/SkySat (optional) | 3-5 m or finer | High revisit (product-dependent) | Global | Commercial API | Paid | Very high spatial detail for targeted studies | Cost/licensing/operational complexity |

## Recommended Tiered Stack

### Tier 1 (Immediate, Open)
- VIIRS 375 m dynamic fire signals.
- 3DEP 10 m terrain features aggregated to 375 m.
- LANDFIRE + NLCD 30 m static features aggregated to 375 m.
- PRISM and/or Daymet daily weather aligned to model timesteps.

### Tier 2 (Retrospective Quality Upgrade)
- HLS 30 m spectral indices (vegetation/water/burn proxy features).
- Sentinel-2/Landsat scene-derived features in selected windows.

### Tier 3 (Optional Paid Enhancement)
- Planet high-resolution imagery for specific fire case studies where sub-10 m scene detail changes conclusions.

## Integration Notes for This Repository

### Current scripts that remain baseline-compatible
- `scripts/ee_download_confidence_stack.py`
- `scripts/ee_download_rtma.py`
- `scripts/run_locational_regressions.py`
- `scripts/run_locational_regressions_aggregate.py`

### New multiresolution path
Use a versioned manifest (see `docs/multires_data_contract.md`) produced/validated by:

- `scripts/build_multires_manifest.py`
- `scripts/validate_multires_manifest.py`

This keeps existing single-resolution scripts unchanged while enabling staged migration.

## Evaluation Protocol
For each candidate tier, run ablations against baseline:

1. Baseline: GOES + RTMA only.
2. Baseline + VIIRS 375 m dynamic labels/features.
3. Baseline + high-resolution static context.
4. Baseline + higher-resolution daily meteorology.
5. Full multires stack.

Track at minimum:
- AUC (existing metric continuity).
- Spatial localization quality (qualitative overlays + quantitative neighborhood hit metrics).
- Calibration drift by fire and by fuel/terrain strata.

## Primary Source Links
- FIRMS: https://firms.modaps.eosdis.nasa.gov
- HLS: https://www.earthdata.nasa.gov/data/projects/hls
- Sentinel-2: https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-2
- Landsat Collection 2: https://www.usgs.gov/landsat-missions/landsat-collection-2
- PRISM: https://prism.oregonstate.edu
- Daymet: https://daymet.ornl.gov
- MRMS: https://www.nssl.noaa.gov/projects/mrms/
- 3DEP: https://www.usgs.gov/3d-elevation-program
- LANDFIRE: https://www.landfire.gov
- NLCD: https://www.usgs.gov/centers/eros/science/national-land-cover-database
