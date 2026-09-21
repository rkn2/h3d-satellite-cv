# h3d-satellite-cv

A pipeline for building a per-building feature inventory from imagery and
public records after a disaster. Given a set of building locations, it pulls
Street View photos, aerial imagery, assessor data, and heritage records, then
classifies each building's structural characteristics using a vision model.

The output is a CSV where every row is a building and every column is a
feature — construction type, roof shape, number of stories, year built,
wall material, damage state, EF rating, historic status, and more. The
current St. Louis tornado dataset has 4,813 buildings and 166 features.

Split from [rkn2/steer](https://github.com/rkn2/steer) (2026-09-21).

---

## How it works

The pipeline has four stages. Each stage adds columns to the building
inventory. You don't need to run all four — each stage's output is useful
on its own.

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  STAGE 1: Image Acquisition                                         │
│                                                                     │
│  Building locations (lat/lon)                                       │
│       │                                                             │
│       ├──→ Google Street View API ──→ 2 facade photos per building  │
│       │    (fetch_streetview.py)       (heading 0° and 90°)         │
│       │                                                             │
│       └──→ Nearmap Vertical API ───→ pre-event + post-event aerial  │
│            (fetch_nearmap.py)         crop per building              │
│                                                                     │
│  What you need: API keys (~/.google_streetview_key, ~/.nearmap_apikey)
│  What you get:  images/streetview/bld_XXXXX/h0.jpg, h90.jpg         │
│                 images/nearmap_cropped/bld_XXXXX_pre.jpg, _post.jpg  │
│                                                                     │
│  Why these sources:                                                  │
│  - Street View gives you the FACADE (wall material, windows, doors, │
│    stories, porch, garage — things you can't see from above)         │
│  - Nearmap gives you the ROOF (shape, covering, damage — and the    │
│    pre/post pair lets you see what changed in the disaster)          │
│  - Together they cover a building from all angles without visiting   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  STAGE 2: Computer Vision Classification                            │
│                                                                     │
│  CLIP (Contrastive Language-Image Pretraining) is a vision model    │
│  that understands both images and text. We use it two ways:         │
│                                                                     │
│  A) SUPERVISED (classify_buildings.py)                              │
│     - Extract a 512-dim embedding from each Street View photo       │
│     - Train gradient boosting classifiers on buildings where a      │
│       human already labeled the answer (the undergrad survey)       │
│     - Predict for all other buildings                               │
│     - Features: stories, roof shape, construction type              │
│     - From these → derive the Memari archetype (T6/T19)             │
│                                                                     │
│  B) ZERO-SHOT (extract_features.py)                                 │
│     - Write text descriptions of each class ("a building with a     │
│       blank brick party wall on the side")                          │
│     - CLIP scores how well each image matches each description      │
│     - No training data needed — but less reliable                   │
│     - Features: shared wall, parapet, porch, garage, foundation,    │
│       wall cladding detail, roof cover type, roof condition,        │
│       roof damage, debris                                           │
│                                                                     │
│  Caution: some zero-shot features don't work well (see below)       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  STAGE 3: Public Records Integration                                │
│                                                                     │
│  For each building, pull data from public databases:                │
│                                                                     │
│  County Assessor (fetch_assessor.py)                                │
│     - Spatial query: send lat/lon → get the parcel it sits on       │
│     - Returns: year built, sq ft, units, zoning, value, owner       │
│     - St. Louis County uses ArcGIS (free, no key needed)            │
│                                                                     │
│  City Assessor (merge_assessor.py, merge_dbf_fields.py)             │
│     - Download the city's parcel CSV (PAR2025) and DBF files        │
│     - Match by address or by spatial parcel polygon join             │
│     - Returns: year built, sq ft, wall code, story code,            │
│       occupancy, historic district, owner-occupied status            │
│     - St. Louis City publishes these free at stlouis-mo.gov/data    │
│                                                                     │
│  NRHP (add_heritage_features.py)                                    │
│     - Query the National Register of Historic Places via NPS API    │
│     - Match buildings within 50m of NRHP-listed properties          │
│     - Returns: listing name, date listed, historic district          │
│                                                                     │
│  NWS Damage Polygons (for tornado events)                           │
│     - Download EF-rated damage survey polygons from the NWS DAT     │
│     - Point-in-polygon join → each building gets an EF rating       │
│     - API: services.dat.noaa.gov (free, no key)                     │
│                                                                     │
│  Heritage Features (add_heritage_features.py)                       │
│     - Derive ownership type from assessor names (individual/        │
│       business/government/religious)                                 │
│     - Derive construction material flags from wall codes            │
│     - Derive timeline (existed before tornado, from year built)     │
│     - Match property type from occupancy data                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  STAGE 4: Quality Control and Filtering                             │
│                                                                     │
│  - Remove post-1950 buildings (not historic)                        │
│  - Remove vacant lots (no building present)                         │
│  - Fill missing year_built from multiple sources                    │
│    (undergrad survey → assessor → web lookup)                       │
│  - Cross-check CLIP predictions against assessor data               │
│    (e.g., assessor story code vs CLIP-predicted stories)            │
│  - Flag unreliable zero-shot features                               │
│                                                                     │
│  Output: building_features.csv (one row per building, 166 columns)  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Current dataset: 2025 St. Louis EF3 Tornado

4,813 pre-1950 buildings along the May 16, 2025 tornado path.

| Source | Buildings | What it provides |
|---|---|---|
| StEER rapid assessment | ~3,750 | Damage rating (5 levels), occupancy |
| Undergrad field survey | ~560 | Archetype, stories, construction, roof, walls, damage (0-5) |
| CLIP supervised classifiers | ~4,700 | Predicted stories, roof shape, construction type, damage |
| CLIP zero-shot | ~4,700 | Shared wall, parapet, foundation, wall cladding, roof features |
| City/County assessor | ~4,500 | Year built (median 1910), sqft, wall code, story code, units, zoning |
| NRHP database | ~1,400 | Historic district designation, listing name |
| NWS DAT polygons | ~4,800 | EF0-EF3 zone assignment |

## Known limitations

**Zero-shot CLIP features that don't work well** (don't use as model inputs):

| Feature | Problem |
|---|---|
| `garage_present` | 85% "yes" — misclassifies driveways and dark openings |
| `porch_present` | 0.1% "yes" — model doesn't detect porches |
| `parapet_present` | 86% "yes" — confuses flat roofline angles with parapets |
| `fenestration_level` | 94% "low" — no discrimination between buildings |
| `roof_cover_type` | 73% "flat_membrane" — aerial perspective confuses the model |

**Zero-shot features that do work:**
`shared_wall`, `wall_cladding_detail`, `debris_present`, `foundation_type`

**Assessor wall codes** (city of St. Louis): numeric codes, not official lookup.
Our best interpretation from cross-referencing with building age and CLIP:
31 = common brick, 32 = wood frame, 42 = stone/larger brick, 43 = mixed/veneer.

## Scripts

Run order for a new event (not all required — each stage is independent):

| Stage | Script | What it does |
|---|---|---|
| 1 | `fetch_streetview.py` | Pull Google Street View facade photos |
| 1 | `fetch_nearmap.py` | Pull Nearmap pre/post aerial crops |
| 2 | `classify_buildings.py` | CLIP embeddings → classifiers → archetype |
| 2 | `extract_features.py` | CLIP zero-shot feature classification |
| 3 | `fetch_assessor.py` | Spatial query to county assessor ArcGIS |
| 3 | `merge_assessor.py` | Address-match to city assessor CSV |
| 3 | `merge_dbf_fields.py` | Extract stories, occupancy from city DBF |
| 3 | `add_heritage_features.py` | NRHP matching, owner classification, heritage flags |
| 4 | (manual / notebook) | Filter, cross-check, fill gaps |

## Data files

| File | Rows | Cols | What |
|---|---|---|---|
| `building_features.csv` | 4,813 | 166 | Everything merged — the main output |
| `buildings_for_incore.csv` | 4,477 | 34 | Subset with archetype + damage + EF for fragility |
| `unified_buildings.csv` | 4,813 | 20 | Original StEER + undergrad merge (Stage 0 input) |
| `classified_buildings.csv` | — | 22 | CLIP predictions only |
| `incore_comparison.csv` | 7 | 17 | IN-CORE fragility comparison results |
| `building-map.html` | — | — | Interactive map (open in browser) |

## Applying to a new event

1. Start with a CSV of building locations (`latitude`, `longitude`, plus
   any existing survey data). This is your `unified_buildings.csv`.
2. Get API keys: Google Street View (`~/.google_streetview_key`) and
   Nearmap (`~/.nearmap_apikey`). Both are paid services.
3. Run Stage 1 scripts to acquire imagery.
4. Run Stage 2 to classify buildings. If you have labeled training data
   (like the undergrad survey), the supervised classifiers will be more
   accurate. Without training data, you still get zero-shot features.
5. Find the local assessor's data portal. Most US counties publish parcel
   data as ArcGIS services, CSVs, or shapefiles. Adapt `fetch_assessor.py`
   and `merge_assessor.py` for the local API.
6. Run `add_heritage_features.py` — the NRHP query works nationally.
7. For tornado events, download NWS DAT polygons for EF assignment.

## How this compares to the satellite pipeline

[rkn2/multihazard-satellite](https://github.com/rkn2/multihazard-satellite)
uses satellite imagery (Maxar, NAIP) with an xView2 change-detection model
to detect **damage** — it tells you whether a building is intact, damaged,
or destroyed, but nothing about the building itself.

This repo (h3d-satellite-cv) uses ground-level Street View + aerial
imagery with CLIP to classify **building characteristics** — it tells you
what the building is made of, how it's built, and how old it is, in
addition to damage.

| | multihazard-satellite | h3d-satellite-cv |
|---|---|---|
| Image source | Satellite (0.3-0.5m, nadir) | Street View (ground) + aerial |
| Model | xView2 (change detection) | CLIP (image-text matching) |
| Detects | Damage only | Building features + damage |
| Sees | Roofs | Facades + roofs |
| Training data | xBD labeled disaster imagery | Small labeled building survey |
| Best for | "Is this building damaged?" | "What is this building and how did it perform?" |

They're complementary: satellite tells you what happened, Street View
tells you what was there.

## Related repos

- [rkn2/steer](https://github.com/rkn2/steer) — event data, reconnaissance dashboards
- [rkn2/multihazard-satellite](https://github.com/rkn2/multihazard-satellite) — satellite damage detection
- [YishuangW1/Incore_Feature_Difference](https://github.com/YishuangW1/Incore_Feature_Difference) (`stl-incore-comparison` branch) — tornado fragility analysis using this dataset
- [YishuangW1/bayesian_update_incore](https://github.com/YishuangW1/bayesian_update_incore) — Bayesian fragility updating

## Requirements

```
pip install open-clip-torch torch pillow pandas numpy scipy scikit-learn shapely requests dbfread
```
