"""Merge assessor parcel data into building_features.csv.

Matches buildings to St. Louis City (by address) and County (by spatial query cache).
"""
import csv
import json
import re

import numpy as np
import pandas as pd

SCRATCH = "/private/tmp/claude-502/-Users-becca-Code-disasters-imageCV/17aa8ff0-86dd-49a1-9843-b102d149c4aa/scratchpad"
FEATURES_PATH = "/Users/becca/Code/disasters/imageCV/building_features.csv"
UNIFIED_PATH = "/Users/becca/Code/disasters/imageCV/unified_buildings.csv"
CITY_CSV = f"{SCRATCH}/stl_city_parcels/Parcels-CSV-2017-2025/PAR2025.csv"
COUNTY_CACHE = f"{SCRATCH}/county_parcels_cache.json"

STREET_ABBREV = {
    "AVENUE": "AV", "AVE": "AV",
    "BOULEVARD": "BL", "BLVD": "BL",
    "CIRCLE": "CIR",
    "COURT": "CT",
    "DRIVE": "DR",
    "HIGHWAY": "HWY",
    "LANE": "LN",
    "PARKWAY": "PY", "PKWY": "PY",
    "PLACE": "PL",
    "ROAD": "RD",
    "STREET": "ST",
    "TERRACE": "TER",
    "WAY": "WY",
}

DIRECTION_ABBREV = {
    "NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W",
    "NORTHEAST": "NE", "NORTHWEST": "NW",
    "SOUTHEAST": "SE", "SOUTHWEST": "SW",
}


def normalize_address(addr):
    if not addr or not isinstance(addr, str) or addr.strip().lower() == "nan":
        return None

    addr = re.sub(r'[^\x00-\x7F]', ' ', addr)
    addr = addr.upper().strip()
    addr = re.sub(r'\s*(,\s*|\s+)(SAINT LOUIS|ST\.?\s*LOUIS|MO|US|UNITED STATES)\b.*', '', addr)
    addr = re.sub(r'\s+\d{5}(-\d{4})?\s*$', '', addr)
    addr = re.sub(r'[,.]', ' ', addr)

    parts = addr.split()
    if not parts:
        return None

    num_part = parts[0]
    if '/' in num_part:
        num_part = num_part.split('/')[0]
    if '-' in num_part and num_part[0].isdigit():
        num_part = num_part.split('-')[0]

    rest = parts[1:]
    normalized = []
    for word in rest:
        word = word.strip()
        if not word:
            continue
        if word in DIRECTION_ABBREV:
            normalized.append(DIRECTION_ABBREV[word])
        elif word in STREET_ABBREV:
            normalized.append(STREET_ABBREV[word])
        else:
            normalized.append(word)

    return f"{num_part} {' '.join(normalized)}".strip()


def normalize_city_addr(addr):
    if not addr or not isinstance(addr, str):
        return None
    addr = addr.upper().strip()
    addr = re.sub(r'[,.]', ' ', addr)

    parts = addr.split()
    if not parts:
        return None

    num_part = parts[0]
    if '-' in num_part and num_part[0].isdigit():
        num_part = num_part.split('-')[0]

    rest = parts[1:]
    normalized = []
    for word in rest:
        word = word.strip()
        if not word:
            continue
        if word in DIRECTION_ABBREV:
            normalized.append(DIRECTION_ABBREV[word])
        elif word in STREET_ABBREV:
            normalized.append(STREET_ABBREV[word])
        else:
            normalized.append(word)

    return f"{num_part} {' '.join(normalized)}".strip()


def main():
    print("Loading data...")
    features = pd.read_csv(FEATURES_PATH)
    unified = pd.read_csv(UNIFIED_PATH)
    city = pd.read_csv(CITY_CSV, low_memory=False)

    with open(COUNTY_CACHE) as f:
        county_cache = json.load(f)

    print(f"  Features: {len(features)} rows, {len(features.columns)} cols")
    print(f"  Unified:  {len(unified)} rows")
    print(f"  City parcels: {len(city)} rows")
    print(f"  County cache: {len(county_cache)} entries, {sum(1 for v in county_cache.values() if v is not None)} matched")

    # Build city parcel lookup
    print("\nBuilding city parcel lookup...")
    city_lookup = {}
    city_by_num_street = {}

    for _, row in city.iterrows():
        raw = str(row.get("SITEADDR", ""))
        norm = normalize_city_addr(raw)
        if norm:
            city_lookup[norm] = row
            parts = norm.split(None, 1)
            if len(parts) == 2 and parts[0].isdigit():
                num = parts[0]
                street = parts[1]
                key = (num, street)
                if key not in city_by_num_street:
                    city_by_num_street[key] = row

    print(f"  City lookup: {len(city_lookup)} normalized addresses")

    # Match buildings
    print("\nMatching buildings...")
    assessor_cols = [
        "assessor_year_built", "assessor_sqft", "assessor_units", "assessor_bldgs",
        "assessor_wall_code", "assessor_land_area", "assessor_frontage",
        "assessor_zoning", "assessor_owner", "assessor_value",
        "assessor_vacant", "assessor_source",
    ]
    for col in assessor_cols:
        features[col] = pd.Series([None] * len(features), dtype=object)

    county_matched = 0
    city_matched = 0
    unmatched = 0

    for i in range(len(features)):
        idx_str = str(i)

        # Try county first
        if idx_str in county_cache and county_cache[idx_str] is not None:
            c = county_cache[idx_str]
            features.at[i, "assessor_year_built"] = c.get("YEARBLT")
            features.at[i, "assessor_sqft"] = c.get("RESQFT")
            features.at[i, "assessor_units"] = c.get("LIVUNIT")
            features.at[i, "assessor_land_area"] = c.get("ACRES")
            features.at[i, "assessor_zoning"] = c.get("ZONING")
            features.at[i, "assessor_owner"] = c.get("OWNER_NAME")
            features.at[i, "assessor_value"] = c.get("TOTAPVAL")
            features.at[i, "assessor_source"] = "county"
            county_matched += 1
            continue

        # Try city by address
        addr = str(unified.at[i, "complete_address"]) if pd.notna(unified.at[i, "complete_address"]) else None
        norm = normalize_address(addr) if addr else None

        match = None
        if norm:
            if norm in city_lookup:
                match = city_lookup[norm]
            else:
                parts = norm.split(None, 1)
                if len(parts) == 2 and parts[0].isdigit():
                    key = (parts[0], parts[1])
                    if key in city_by_num_street:
                        match = city_by_num_street[key]

        if match is not None:
            features.at[i, "assessor_year_built"] = match.get("BDG1YEAR")
            features.at[i, "assessor_sqft"] = match.get("BDG1AREA")
            features.at[i, "assessor_units"] = match.get("NUMUNITS")
            features.at[i, "assessor_bldgs"] = match.get("NUMBLDGS")
            features.at[i, "assessor_wall_code"] = match.get("BDG1EXWALL")
            features.at[i, "assessor_land_area"] = match.get("LANDAREA")
            features.at[i, "assessor_frontage"] = match.get("FRONTAGE")
            features.at[i, "assessor_zoning"] = match.get("ZONING1")
            features.at[i, "assessor_owner"] = match.get("OWNERNAME")
            features.at[i, "assessor_value"] = match.get("ASMTTOTAL")
            features.at[i, "assessor_vacant"] = match.get("VACANTLAND")
            features.at[i, "assessor_source"] = "city"
            city_matched += 1
        else:
            unmatched += 1

    print(f"\n=== Results ===")
    print(f"County matched: {county_matched}")
    print(f"City matched:   {city_matched}")
    print(f"Unmatched:      {unmatched}")
    print(f"Total:          {len(features)}")
    print(f"Match rate:     {100*(county_matched+city_matched)/len(features):.1f}%")

    # Field coverage
    print(f"\n=== Field coverage (matched buildings) ===")
    matched = features[features.assessor_source.notna()]
    for col in assessor_cols:
        if col == "assessor_source":
            continue
        nn = matched[col].notna().sum()
        # Filter out 0 values for year and sqft
        if col in ("assessor_year_built", "assessor_sqft"):
            nonzero = ((matched[col].notna()) & (matched[col] != 0)).sum()
            print(f"  {col:25s}: {nn:5d} filled, {nonzero:5d} non-zero")
        else:
            print(f"  {col:25s}: {nn:5d} filled")

    # Sample matches
    print(f"\n=== Sample city matches ===")
    city_rows = features[features.assessor_source == "city"].head(5)
    for _, row in city_rows.iterrows():
        print(f"  yr={row.assessor_year_built} sqft={row.assessor_sqft} "
              f"units={row.assessor_units} wall={row.assessor_wall_code} "
              f"zone={row.assessor_zoning} vacant={row.assessor_vacant}")

    print(f"\n=== Sample county matches ===")
    county_rows = features[features.assessor_source == "county"].head(5)
    for _, row in county_rows.iterrows():
        print(f"  yr={row.assessor_year_built} sqft={row.assessor_sqft} "
              f"units={row.assessor_units} zone={row.assessor_zoning} "
              f"value={row.assessor_value}")

    # Year built distribution for matched
    yr = matched["assessor_year_built"]
    yr = yr[(yr.notna()) & (yr > 0)]
    if len(yr) > 0:
        print(f"\n=== Year built distribution ===")
        print(f"  min={yr.min():.0f}, median={yr.median():.0f}, max={yr.max():.0f}")
        bins = [0, 1900, 1920, 1940, 1960, 1980, 2000, 2030]
        labels = ["<1900", "1900-19", "1920-39", "1940-59", "1960-79", "1980-99", "2000+"]
        yr_binned = pd.cut(yr, bins=bins, labels=labels)
        print(yr_binned.value_counts().sort_index().to_string())

    # Save
    features.to_csv(FEATURES_PATH, index=False)
    print(f"\nSaved to {FEATURES_PATH}: {len(features)} rows, {len(features.columns)} cols")


if __name__ == "__main__":
    main()
