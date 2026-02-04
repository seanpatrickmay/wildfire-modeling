# Earth Engine Scripts

This folder contains the Earth Engine JavaScript pipeline used to build the GOFER product. Files are numbered to match the workflow.

## Workflow summary

- 0a - Calc_stagingAOI.js: set temporal and spatial constraints for each fire.
- 0b - Calc_KernelRes.js: compute smoothing kernel radius based on GOES resolution.
- 0c - Export_FireData.js: export fire metadata.
- 1a - Export_FireConf.js: export GOES fire detection confidence.
- 1b - Export_Parallax.js: export parallax displacement in x/y.
- 2 - Export_ParamSens.js: optimize confidence thresholds and parallax adjustments.
- 3 - Export_ScaleVal.js: export early perimeter scaling factors.
- 4 - Export_FireProg.js: export fire perimeters.
- 5 - Export_FireProgQA.js: quality control post-processing.
- 6a - Export_cFireLine.js: export concurrent active fire lines.
- 6b - Export_rFireLine.js: export retrospective active fire lines.
- 6c - Export_FireIg.js: export fire ignitions.
- 7 - Export_FireProgStats.js: export fire spread rate and growth stats.

## Earth Engine assets

Typical GOFER assets include:

- GOESEast_MaxConf and GOESWest_MaxConf
- GOESEast_Parallax and GOESWest_Parallax
- GOFERC_fireProg, GOFERC_fireIg, GOFERC_cfireLine, GOFERC_rfireLine

See the root README for the full asset tree.
