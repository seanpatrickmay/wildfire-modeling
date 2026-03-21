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
- `docs/neighbor_cell_confidence_regression.ipynb` (single-fire + multi-fire logistic blocks)

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

---

## Supplementary Data Sources (Research Update, March 2026)

The sections below document additional data sources researched as supplements or
alternatives to the current RTMA + GOES + VIIRS + RAWS baseline. Each entry
includes resolution details, access methods, advantages relative to the current
stack, integration complexity, and references.

### HRRR (High-Resolution Rapid Refresh)

| Attribute | Detail |
|---|---|
| **Spatial resolution** | 3 km |
| **Temporal resolution** | Hourly forecasts (18 h every hour, 48 h every 6 h); 15-min radar assimilation cycle |
| **Coverage** | CONUS |
| **Record** | Archive from 2014 to present |
| **Data access** | AWS S3 (`s3://noaa-hrrr-bdp-pds`, Zarr archive at `s3://hrrrzarr`); Google Cloud Platform (`gs://high-resolution-rapid-refresh`); **not natively in GEE** -- requires external ingest. Python access via the [Herbie](https://herbie.readthedocs.io/) library. |
| **License/cost** | Open (NOAA public domain) |

**Comparison with RTMA.** RTMA is an *analysis* product (observation-assimilated gridded field), while HRRR is a *forecast model*. Critically, RTMA uses HRRR output as its background field, so HRRR and RTMA share the same 3 km native grid. The key differences are:

- HRRR provides forecast lead times (1-48 h), enabling predictive fire weather features; RTMA is analysis-only (t=0).
- HRRRv4 includes explicit wildfire smoke plume prediction, useful for smoke-aware modeling.
- HRRR carries full 3-D atmospheric fields (e.g., vertical profiles of wind, moisture) not available in surface-only RTMA.
- RTMA has the advantage of being constrained more tightly to surface observations via its data assimilation step.

**Key advantages over current data.** Forecast capability for predictive features; smoke prediction; full atmospheric column; same spatial resolution as RTMA but with temporal forecast dimension.

**Integration complexity.** Medium-high. HRRR output is in GRIB2 on a Lambert conformal grid. Ingesting into the existing GEE-based pipeline requires either (a) converting GRIB2 to GeoTIFF and uploading as GEE assets, or (b) building a parallel pipeline that reads directly from S3/GCS using `xarray` + `cfgrib` or Herbie. The Zarr archive on AWS simplifies bulk historical access.

**References.**
- James et al., 2022: "The High-Resolution Rapid Refresh (HRRR)." *Weather and Forecasting*, 37(8). https://journals.ametsoc.org/view/journals/wefo/37/8/WAF-D-21-0151.1.xml
- AWS registry: https://registry.opendata.aws/noaa-hrrr-pds/
- Herbie docs: https://herbie.readthedocs.io/en/stable/user_guide/background/model-info/hrrr.html

---

### ERA5-Land (ECMWF Reanalysis)

| Attribute | Detail |
|---|---|
| **Spatial resolution** | ~9 km (0.1 degree) |
| **Temporal resolution** | Hourly |
| **Coverage** | Global land |
| **Record** | 1950 to ~5 days before present |
| **Data access** | **Google Earth Engine** (`ECMWF/ERA5_LAND/HOURLY`, also daily and monthly aggregates); Copernicus Climate Data Store (CDS) API; AWS (via ECMWF open data). |
| **License/cost** | Open (Copernicus licence); free via GEE |

**Comparison with RTMA.** ERA5-Land is coarser (9 km vs 2.5 km) but provides three decisive advantages:

- **76-year historical record** (1950-present) vs RTMA's ~2012-present archive, enabling long-term climatological fire-weather analysis.
- **Global consistency** -- no missing hours, no station dropout, no edge effects. Reanalysis fills all gaps by design.
- **50 surface variables** including soil moisture (4 layers), snow, lake temperature, and full radiation budget, providing land-surface-state context that RTMA lacks.

ERA5-Land is less suited as a replacement for RTMA in near-real-time applications due to its ~5-day latency, but for retrospective modeling and training data construction it is superior for consistency and variable breadth.

**Key advantages over current data.** Very long consistent historical record; global coverage; rich soil/land-surface variables; native GEE availability simplifies integration with existing EE-based scripts.

**Integration complexity.** Low. Already available in GEE with the same API patterns used by `ee_download_rtma.py`. A new download script modeled on the RTMA downloader could be built quickly. Main consideration: the 9 km resolution is coarser than the 2.5 km RTMA, so it supplements rather than replaces RTMA for spatial detail.

**References.**
- Munoz-Sabater et al., 2021: "ERA5-Land: a state-of-the-art global reanalysis dataset for land applications." *Earth System Science Data*, 13, 4349-4383. https://essd.copernicus.org/articles/13/4349/2021/
- GEE catalog: https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_HOURLY
- ERA5-based fire danger maps: Vitolo et al., 2020. https://www.nature.com/articles/s41597-020-0554-z

---

### LANDFIRE Fuel Models (Expanded Detail)

The existing document lists LANDFIRE at 30 m. This section adds operational detail
for fuel model integration.

| Attribute | Detail |
|---|---|
| **Spatial resolution** | 30 m raster |
| **Temporal resolution** | Static per release cycle; LF 2024 (v2.5.0) is the latest, releasing through 2025 |
| **Coverage** | CONUS, Alaska, Hawaii, Puerto Rico |
| **Data access** | Direct download from https://www.landfire.gov/data; **Google Earth Engine** community catalog (`projects/sat-io/open-datasets/landfire/FUEL/FBFM40`, etc.); USGS data catalog |
| **License/cost** | Public domain, no restrictions |

**Key fuel products for fire spread modeling.**

| Product | Code | Use |
|---|---|---|
| Scott & Burgan 40 Fire Behavior Fuel Models | FBFM40 | Primary input for FARSITE, FlamMap, and similar operational spread models; assigns rate-of-spread and flame-length parameters |
| Anderson 13 Fire Behavior Fuel Models | FBFM13 | Simpler 13-class system compatible with Rothermel spread equations |
| Canopy Bulk Density | CBD | Crown fire potential |
| Canopy Base Height | CBH | Crown fire initiation threshold |
| Canopy Cover | CC | Shading, wind reduction, spotting |
| Forest Canopy Fuel Stratum Characteristics | FCCS | Detailed fuel stratum structure |

**Integration with fire spread models.** LANDFIRE FBFM40 is the standard fuel input for all major operational fire spread simulators (FARSITE, FlamMap, BEHAVE, Prometheus). For ML-based models, fuel model classes can be one-hot encoded or embedded, or fuel physical properties (fuel load, surface-area-to-volume ratio, moisture of extinction) can be looked up from the Scott & Burgan fuel model tables and used as continuous features.

**Calibration note.** LANDFIRE regularly calibrates fuel rulesets against observed fire behavior. The most recent calibrations (Colorado and Southeast GeoAreas) were completed December 2024 for the LF 2023 update, with LF 2024 calibrations ongoing. Users should verify that their fuel layer version matches the time period being modeled.

**MoD-FIS (Modeling Dynamic Fuels with an Index System).** LANDFIRE's MoD-FIS product adjusts static fuel models by disturbance history, providing a more temporally relevant fuel assignment. Available for CONUS.

**Key advantages over current data.** Directly parameterizes fire spread physics; 30 m resolution captures fuel heterogeneity at landscape scale; GEE availability enables integration without file management.

**Integration complexity.** Low-medium. For the ML pipeline: download or query FBFM40 in GEE, aggregate to the model grid (375 m), and include as categorical or physical-property features. For physics-based spread models: FBFM40 is the expected native input format and requires no transformation.

**References.**
- LANDFIRE 2024 Update: https://www.landfire.gov/data/lf2024
- LF fuel products in GEE community catalog: https://gee-community-catalog.org/projects/landfire/
- Scott & Burgan, 2005: "Standard Fire Behavior Fuel Models." USDA Forest Service RMRS-GTR-153.

---

### SRTM / ASTER DEM (Terrain Predictors)

The existing document covers 3DEP (10/30 m, CONUS only). SRTM and ASTER provide
complementary global-coverage DEMs relevant if the project expands beyond CONUS.

| Attribute | SRTM V3 | ASTER GDEM V3 |
|---|---|---|
| **Spatial resolution** | 30 m (1 arc-sec) or 90 m (3 arc-sec) | 30 m (1 arc-sec) |
| **Coverage** | 60N-56S latitude | Global land |
| **Data access** | **GEE**: `USGS/SRTMGL1_003` (30 m), `CGIAR/SRTM90_V4` (90 m); NASA Earthdata; OpenTopography | **GEE**: `NASA/ASTER_GED/AG100_003`; NASA Earthdata |
| **License/cost** | Open (public domain) | Open |

**Terrain variables as fire behavior predictors.** Slope, aspect, and elevation are
fundamental inputs to all fire spread models:

- **Slope** directly modifies head-fire rate of spread (Rothermel equation); fires accelerate dramatically on steep uphill terrain.
- **Aspect** controls solar exposure and fuel drying. South-facing slopes (Northern Hemisphere) receive more insolation, resulting in drier, more combustible fuels.
- **Elevation** influences temperature, humidity, fuel type, and snow cover timing.
- **Derived indices** such as Topographic Position Index (TPI), Continuous Heat-Insolation Load Index (CHILI, available in GEE as `CSP/ERGo/1_0/Global/SRTM_CHILI`), and topographic diversity add further predictive value.

**SRTM vs 3DEP.** For CONUS work, 3DEP is preferred (higher accuracy, lidar-derived in many areas). SRTM/ASTER are relevant if the project extends globally or needs a quick global baseline. Note that SRTM accuracy degrades on steep slopes and is influenced by aspect-dependent biases.

**Key advantages over current data.** Terrain is not currently used as a predictor in the regression pipeline. Adding slope, aspect, and TPI at 30 m (aggregated to model grid) would capture a major fire behavior driver at minimal data cost. GEE has SRTM ready with built-in `ee.Terrain.products()` for slope/aspect computation.

**Integration complexity.** Very low. A few lines of GEE code produce slope, aspect, and elevation layers aligned to the model grid. Can be added to the existing `ee_download_confidence_stack.py` or a new static-context download script.

**References.**
- NASA SRTM in GEE: https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003
- CGIAR SRTM 90m: https://developers.google.com/earth-engine/datasets/catalog/CGIAR_SRTM90_V4
- SRTM CHILI in GEE: https://developers.google.com/earth-engine/datasets/catalog/CSP_ERGo_1_0_Global_SRTM_CHILI
- McRae, 2021: "Modelling terrain for wildfire purposes." MODSIM2021.

---

### GridMET (Gridded Surface Meteorological Data)

| Attribute | Detail |
|---|---|
| **Spatial resolution** | ~4 km (1/24 degree) |
| **Temporal resolution** | Daily |
| **Coverage** | CONUS |
| **Record** | 1979 to yesterday (updated daily) |
| **Data access** | **Google Earth Engine** (`IDAHO_EPSCOR/GRIDMET`); Climatology Lab downloads; Microsoft Planetary Computer; Climate Engine |
| **License/cost** | Open |

**Key variables for fire weather.** GridMET includes standard meteorological fields
(temperature, precipitation, humidity, wind, radiation) plus pre-computed fire
danger indices that are not available in RTMA:

- **Energy Release Component (ERC)** -- cumulative drying metric widely used by fire management to track fire season severity.
- **Burning Index (BI)** -- combines spread rate and energy release.
- **100-hour and 1000-hour dead fuel moisture** -- key inputs to fire behavior calculations.
- **Fire Danger indices** based on the National Fire Danger Rating System (NFDRS).

**Comparison with RTMA.** GridMET is coarser (4 km vs 2.5 km) and daily-only (vs hourly RTMA). However, GridMET provides fire-specific derived variables (ERC, fuel moistures) that would need to be computed separately from RTMA fields. GridMET blends PRISM spatial climatology with NLDAS-2 temporal patterns, producing a validated, gap-free daily product back to 1979 -- far longer than RTMA's record.

**Key advantages over current data.** Pre-computed fire danger indices (ERC, BI, fuel moisture); long historical record (1979+); native GEE availability; daily updates; validated against weather station networks.

**Integration complexity.** Low. Same GEE API as existing scripts. Add `IDAHO_EPSCOR/GRIDMET` as an additional image collection in a new or extended download script. Fire danger bands can be extracted directly without post-processing.

**References.**
- Abatzoglou, 2013: "Development of gridded surface meteorological data for ecological applications and modelling." *International Journal of Climatology*, 33(1), 121-131.
- GEE catalog: https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_GRIDMET
- Climatology Lab: https://www.climatologylab.org/gridmet.html

---

### VIIRS Active Fire Confidence Thresholds (Operational Guidance)

This section supplements the existing FIRMS VIIRS entry with confidence threshold
guidance for fire studies.

**Confidence classification.** VIIRS I-Band 375 m active fire detections are assigned
three confidence classes:

| Class | Confidence range | Characteristics |
|---|---|---|
| **Low** | 0-29% | Often associated with sun glint, low temperature anomaly (<15 K in I4 band), or sub-pixel cloud edges. Higher false alarm rate. |
| **Nominal** | 30-79% | Free of sun-glint contamination; strong temperature anomaly (>15 K) in day or night data. Standard threshold for most studies. |
| **High** | 80-100% | Saturated pixels in day or night; very high certainty of active fire. |

**Standard practice in fire studies.** Most peer-reviewed wildfire studies retain
**nominal + high confidence** detections (>=30%) and discard low-confidence
pixels. This balances false-alarm suppression with detection completeness. Users
requiring maximum sensitivity (e.g., early detection of new ignitions) may include
all three classes but should expect higher commission error.

**Nighttime filtering caveat.** Recent research (2025) has identified that VIIRS
appears to systematically exclude low-confidence nighttime detections below ~295 K
brightness temperature entirely, rather than flagging them as low-confidence. This
means the nighttime detection population is already pre-filtered, and the
low/nominal/high split is not symmetric between day and night. Studies relying on
nighttime fire counts should account for this undocumented algorithmic filtering.

**Recommendation for this project.** Use **confidence >= "nominal"** (>=30%) as the
default filter for VIIRS detections in training data and validation. Document
the threshold in data manifests. Consider separate day/night analysis given the
asymmetric nighttime filtering behavior.

**References.**
- VIIRS I-Band 375 m Active Fire Data: https://www.earthdata.nasa.gov/data/instruments/viirs/viirs-i-band-375-m-active-fire-data
- VIIRS Active Fire Product User's Guide: https://lpdaac.usgs.gov/documents/427/VNP14_User_Guide_V1.pdf
- Nighttime filtering analysis: https://arxiv.org/html/2510.26816v1
- FIRMS FAQ: https://www.earthdata.nasa.gov/data/tools/firms/faq

---

### New Satellite Fire Detection Products (2024-2026)

Several new fire detection systems have emerged since 2024 that may supplement or
eventually supersede GOES-based detection for this project.

#### NOAA Next Generation Fire System (NGFS)

| Attribute | Detail |
|---|---|
| **Platform** | GOES-16, GOES-19 (geostationary, ABI instrument) |
| **Detection cadence** | Every 1 minute (mesoscale sectors), full CONUS every 5 minutes |
| **Minimum fire size** | ~0.25 acres |
| **Latency** | Alerts within 1 minute of energy reaching the satellite |
| **Status** | Experimental; operational transition expected late 2026 |
| **Access** | CIMSS portal (https://cimss.ssec.wisc.edu/ngfs/); NOAA wildfire data portal |
| **Cost** | Open (developed for under $3M total) |

NGFS uses AI-driven algorithms to continuously identify heat anomalies from GOES ABI
data, including detection through clouds and smoke. It represents a major upgrade
over the current GOES fire detection products used in this project. When it reaches
operational status (expected late 2026), it could replace the current GOFER
confidence approach with higher-sensitivity, lower-latency detections.

#### FireSat (Earth Fire Alliance / Google / Muon Space)

| Attribute | Detail |
|---|---|
| **Platform** | Dedicated LEO constellation (Muon Space-built satellites) |
| **Detection resolution** | Fires as small as 5x5 meters |
| **Cadence** | First 3 satellites (mid-2026): twice-daily global; full constellation (50+ satellites by ~2030): every 20 minutes over fire-prone regions |
| **Status** | Protoflight satellite launched March 2025 (SpaceX Transporter-13); first light images confirmed; first 3 operational satellites planned mid-2026 |
| **Access** | TBD; data expected to be open via Earth Fire Alliance |
| **Cost** | Funded by Google.org ($13M+), Moore Foundation, EDF |

FireSat represents a step change in spatial resolution for fire detection (5 m vs
375 m VIIRS / ~2 km GOES). The full constellation's 20-minute revisit would
approach geostationary cadence with far higher spatial precision. Timeline for
research use is likely 2027+ for substantial archives.

**Reference:** https://www.earthfirealliance.org/press-release/firesat-first-wildfire-images

#### OroraTech (Commercial LEO Constellation)

| Attribute | Detail |
|---|---|
| **Platform** | Constellation of miniaturized thermal-IR satellites |
| **Detection speed** | 3-minute on-orbit detection demonstrated |
| **Status** | Operational constellation launching since 2024; also integrating with MeteoSat-12 geostationary data |
| **Access** | Commercial platform (subscription-based) |
| **Cost** | Paid |

OroraTech offers near-real-time thermal monitoring through its own satellite
constellation and by processing data from MeteoSat-12 (which provided 41% of
first detections in European wildfire clusters in August 2025). Primarily relevant
as a commercial operational tool rather than a research data source.

**Reference:** https://ororatech.com/

#### MeteoSat-12 (EUMETSAT, Geostationary)

| Attribute | Detail |
|---|---|
| **Platform** | Geostationary (EUMETSAT) |
| **Scan interval** | Every 10 minutes; 15-minute data latency |
| **Coverage** | Europe, Africa, parts of South America |
| **Status** | Operational; fire detection data available in FIRMS since mid-2024 |
| **Access** | NASA FIRMS; EUMETSAT Data Store |
| **Cost** | Open |

Not directly relevant for CONUS work, but notable as a demonstration of
geostationary fire detection improvements. The FIRMS integration means geostationary
fire detections from GOES, MeteoSat, and Himawari are now available in a unified
format.

**Reference:** https://www.earthdata.nasa.gov/news/blog/geostationary-active-fire-detection-data-firms

---

## Updated Data Source Matrix (Supplementary Sources)

| Source | Native Resolution | Temporal Cadence | Coverage | Access | License/Cost | Strengths | Limitations |
|---|---:|---|---|---|---|---|---|
| HRRR | 3 km | Hourly forecasts | CONUS | AWS S3, GCS (not GEE) | Open | Forecast capability, smoke prediction, same grid as RTMA | GRIB2 format, no GEE, forecast not analysis |
| ERA5-Land | ~9 km | Hourly | Global land | **GEE**, CDS API | Open | 1950-present record, 50 surface variables, gap-free | Coarser than RTMA, ~5-day latency |
| GridMET | ~4 km | Daily | CONUS | **GEE**, downloads | Open | Pre-computed fire danger indices (ERC, BI, fuel moisture), 1979+ | Daily only, coarser than RTMA |
| SRTM DEM | 30 m / 90 m | Static | 60N-56S | **GEE** | Open | Global terrain, slope/aspect in GEE natively | Less accurate than 3DEP for CONUS |
| NGFS | ~2 km (GOES ABI) | 1-5 min | CONUS | CIMSS portal | Open | AI-enhanced detection, sub-acre fires, through-cloud/smoke | Experimental until late 2026 |
| FireSat | 5 m detection | 20 min (full constellation) | Global | TBD (Earth Fire Alliance) | Expected open | Unprecedented spatial resolution for fire detection | Full constellation not until ~2030 |

---

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
- HRRR (AWS): https://registry.opendata.aws/noaa-hrrr-pds/
- HRRR (Herbie): https://herbie.readthedocs.io/en/stable/user_guide/background/model-info/hrrr.html
- ERA5-Land (GEE): https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_HOURLY
- ERA5-Land (ECMWF): https://www.ecmwf.int/en/era5-land
- GridMET (GEE): https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_GRIDMET
- GridMET (Climatology Lab): https://www.climatologylab.org/gridmet.html
- SRTM 30m (GEE): https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003
- NGFS (CIMSS): https://cimss.ssec.wisc.edu/ngfs/
- FireSat: https://www.earthfirealliance.org/press-release/firesat-first-wildfire-images
- OroraTech: https://ororatech.com/
- VIIRS Active Fire User Guide: https://lpdaac.usgs.gov/documents/427/VNP14_User_Guide_V1.pdf
