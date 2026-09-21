# h3d-satellite-cv

Computer vision pipeline for classifying historic building characteristics
from Street View and aerial imagery. Integrates CLIP embeddings, municipal
assessor records, and heritage databases to produce rich building inventories
for post-disaster damage assessment.

Split from [rkn2/steer](https://github.com/rkn2/steer) (2026-09-21,
full history preserved with `git filter-repo`).

## Pipeline

```
Street View images ──→ CLIP embeddings ──→ Supervised classifiers ──→ Archetype
Nearmap aerials ────→ CLIP embeddings ──→ Damage classification     (T6/T19/...)
                                        → Zero-shot features

Addresses ──→ City/County assessor API ──→ Year built, sqft, wall type, units
Lat/lon ───→ Parcel spatial join ────────→ Zoning, owner, value
           → NRHP spatial match ─────────→ Heritage designation
           → NWS DAT polygons ───────────→ EF rating assignment
```

## Current dataset: 2025 St. Louis EF3 Tornado

4,813 pre-1950 buildings, 166 features per building, 99.8% year_built coverage.

| Source | Buildings | Features |
|---|---|---|
| StEER rapid assessment | ~3,750 | Damage rating, occupancy |
| Undergrad field survey | ~560 | Archetype, stories, construction, roof, walls |
| CLIP supervised classifiers | ~4,700 | Predicted stories, roof, construction, damage |
| CLIP zero-shot | ~4,700 | Shared wall, fenestration, parapet, foundation, etc. |
| City assessor (PAR2025 + DBF) | ~4,500 | Year built, sqft, wall code, stories, units, zoning |
| County assessor (ArcGIS) | ~480 | Year built, sqft, value, owner, zoning |
| NRHP database | ~1,400 | Historic district, listing date, property name |
| NWS DAT polygons | ~4,800 | EF0–EF3 assignment |

## Scripts

| Script | What it does |
|---|---|
| `classify_buildings.py` | CLIP ViT-B-32 → gradient boosting classifiers (stories, roof, construction, damage). Derives Memari archetype. |
| `extract_features.py` | Zero-shot CLIP for 12 features (shared wall, parapet, soffit, porch, garage, foundation, wall cladding, roof cover/condition/damage, debris). |
| `fetch_streetview.py` | Google Street View Static API acquisition. |
| `fetch_nearmap.py` | Nearmap Vertical API pre/post aerial acquisition. |
| `merge_assessor.py` | Address-match to St. Louis City PAR2025 CSV. |
| `fetch_assessor.py` | Spatial query vs St. Louis County ArcGIS parcels. |
| `merge_dbf_fields.py` | Stories, occupancy, historic district from city DBF. |
| `add_heritage_features.py` | NRHP matching, owner classification, heritage features. |

## Data files

| File | Rows | Cols | Description |
|---|---|---|---|
| `building_features.csv` | 4,813 | 166 | Master dataset — all features merged |
| `buildings_for_incore.csv` | 4,477 | 34 | IN-CORE comparison input |
| `unified_buildings.csv` | 4,813 | 20 | Original merged inventory |
| `classified_buildings.csv` | — | 22 | CLIP predictions |
| `incore_comparison.csv` | 7 | 17 | Per-stratum RPSS |
| `building-map.html` | — | — | Interactive Leaflet map |

## Related repos

- [rkn2/steer](https://github.com/rkn2/steer) — event data, dashboards, event READMEs
- [YishuangW1/Incore_Feature_Difference](https://github.com/YishuangW1/Incore_Feature_Difference) (`stl-incore-comparison` branch) — IN-CORE comparison, T20 fragility, gap analysis
- [YishuangW1/bayesian_update_incore](https://github.com/YishuangW1/bayesian_update_incore) — Bayesian fragility updating
- [rkn2/multihazard-satellite](https://github.com/rkn2/multihazard-satellite) — xView2 satellite damage detection

## Requirements

```
pip install open-clip-torch torch pillow pandas numpy scipy scikit-learn shapely requests dbfread
```

Google Street View API key in `~/.google_streetview_key`. Nearmap API key in `~/.nearmap_apikey`.

## Applying to a new event

1. Prepare `unified_buildings.csv` with lat/lon + any existing survey data
2. Run `fetch_streetview.py` and `fetch_nearmap.py` to acquire imagery
3. Run `classify_buildings.py` to predict archetypes and damage
4. Run `extract_features.py` for zero-shot features
5. Adapt `merge_assessor.py` / `fetch_assessor.py` for the local assessor API
6. Run `add_heritage_features.py` for NRHP matching
7. Fetch NWS DAT polygons for EF assignment (tornado events)
