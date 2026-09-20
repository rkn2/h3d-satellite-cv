"""Fetch property data from St. Louis County ArcGIS, City parcel DB, and NRHP.

Merges results into building_features.csv.
"""
import csv
import json
import os
import time
import urllib.request
import urllib.parse
import numpy as np

SCRATCH = "/private/tmp/claude-502/-Users-becca-Code-disasters-imageCV/17aa8ff0-86dd-49a1-9843-b102d149c4aa/scratchpad"
BASE = "/Users/becca/Code/disasters/imageCV"

COUNTY_URL = "https://maps.stlouisco.com/hosting/rest/services/Maps/AGS_Parcels/MapServer/0/query"
COUNTY_FIELDS = "YEARBLT,RESQFT,LIVUNIT,PROPCLASS,LANDUSE2,TOTAPVAL,OWNER_NAME,TENURE,ACRES,ZONING,MUNICIPALITY"

NRHP_URL = "https://mapservices.nps.gov/arcgis/rest/services/cultural_resources/nrhp_locations/MapServer/0/query"


def query_county_parcel(lat, lon):
    params = urllib.parse.urlencode({
        "geometry": json.dumps({"x": lon, "y": lat}),
        "geometryType": "esriGeometryPoint",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": COUNTY_FIELDS,
        "inSR": "4326",
        "returnGeometry": "false",
        "f": "json",
    })
    url = f"{COUNTY_URL}?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        features = data.get("features", [])
        if features:
            return features[0]["attributes"]
    except Exception as e:
        pass
    return None


def query_nrhp_bbox(min_lat, min_lon, max_lat, max_lon):
    params = urllib.parse.urlencode({
        "geometry": json.dumps({
            "xmin": min_lon, "ymin": min_lat,
            "xmax": max_lon, "ymax": max_lat
        }),
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "resname,nris_refnum,mult_name,st_NR_date,county,state",
        "inSR": "4326",
        "outSR": "4326",
        "returnGeometry": "true",
        "f": "json",
    })
    url = f"{NRHP_URL}?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        return data.get("features", [])
    except Exception as e:
        print(f"  NRHP query error: {e}")
        return []


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def main():
    # Load buildings
    with open(os.path.join(BASE, "unified_buildings.csv")) as f:
        buildings = list(csv.DictReader(f))
    n = len(buildings)
    print(f"Loaded {n} buildings.")

    # --- Source 1: St. Louis County parcels ---
    print("\n=== Source 1: St. Louis County ArcGIS parcels ===")
    county_results = {}
    county_match = 0
    county_miss = 0

    cache_path = os.path.join(SCRATCH, "county_parcels_cache.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            county_results = json.load(f)
        print(f"  Loaded {len(county_results)} cached results.")

    uncached = [i for i in range(n) if str(i) not in county_results]
    print(f"  Need to query {len(uncached)} buildings...")

    for batch_idx, i in enumerate(uncached):
        lat = float(buildings[i]["latitude"])
        lon = float(buildings[i]["longitude"])
        result = query_county_parcel(lat, lon)
        county_results[str(i)] = result
        if result:
            county_match += 1
        else:
            county_miss += 1

        if (batch_idx + 1) % 100 == 0:
            print(f"  Queried {batch_idx + 1}/{len(uncached)} ({county_match} matches so far)")
            with open(cache_path, "w") as f:
                json.dump(county_results, f)

        time.sleep(0.2)

    with open(cache_path, "w") as f:
        json.dump(county_results, f)

    total_match = sum(1 for v in county_results.values() if v is not None)
    total_miss = sum(1 for v in county_results.values() if v is None)
    print(f"  County results: {total_match} matched, {total_miss} no match (likely city)")

    # --- Source 3: NRHP ---
    print("\n=== Source 3: NRHP ===")
    lats = [float(b["latitude"]) for b in buildings]
    lons = [float(b["longitude"]) for b in buildings]
    min_lat, max_lat = min(lats) - 0.01, max(lats) + 0.01
    min_lon, max_lon = min(lons) - 0.01, max(lons) + 0.01

    nrhp_features = query_nrhp_bbox(min_lat, min_lon, max_lat, max_lon)
    print(f"  NRHP properties in study area: {len(nrhp_features)}")

    nrhp_points = []
    for feat in nrhp_features:
        geom = feat.get("geometry", {})
        attrs = feat.get("attributes", {})
        if geom and "x" in geom and "y" in geom:
            nrhp_points.append({
                "lat": geom["y"], "lon": geom["x"],
                "name": attrs.get("resname", ""),
                "ref_num": attrs.get("nris_refnum", ""),
                "date_listed": attrs.get("st_NR_date", ""),
            })

    nrhp_matches = {}
    for i, bld in enumerate(buildings):
        blat = float(bld["latitude"])
        blon = float(bld["longitude"])
        best_dist = 999999
        best_match = None
        for nrhp in nrhp_points:
            d = haversine_m(blat, blon, nrhp["lat"], nrhp["lon"])
            if d < best_dist:
                best_dist = d
                best_match = nrhp
        if best_dist < 50:
            nrhp_matches[str(i)] = {
                "name": best_match["name"],
                "date_listed": best_match["date_listed"],
                "distance_m": round(best_dist, 1),
            }

    print(f"  NRHP matches (within 50m): {len(nrhp_matches)}")

    # --- Merge into building_features.csv ---
    print("\n=== Merging into building_features.csv ===")
    features_path = os.path.join(BASE, "building_features.csv")
    if os.path.exists(features_path):
        with open(features_path) as f:
            reader = csv.DictReader(f)
            existing_fields = reader.fieldnames
            features = list(reader)
        print(f"  Loaded existing building_features.csv: {len(features)} rows, {len(existing_fields)} cols")
    else:
        with open(os.path.join(BASE, "unified_buildings.csv")) as f:
            reader = csv.DictReader(f)
            existing_fields = reader.fieldnames
            features = list(reader)
        print(f"  No existing features file, using unified_buildings.csv")

    new_fields = [
        "assessor_year_built", "assessor_sqft", "assessor_units",
        "assessor_prop_class", "assessor_land_use", "assessor_value",
        "assessor_owner", "assessor_tenure", "assessor_zoning",
        "assessor_municipality", "assessor_source",
        "nrhp_listed", "nrhp_name", "nrhp_date_listed",
    ]

    field_map = {
        "YEARBLT": "assessor_year_built",
        "RESQFT": "assessor_sqft",
        "LIVUNIT": "assessor_units",
        "PROPCLASS": "assessor_prop_class",
        "LANDUSE2": "assessor_land_use",
        "TOTAPVAL": "assessor_value",
        "OWNER_NAME": "assessor_owner",
        "TENURE": "assessor_tenure",
        "ZONING": "assessor_zoning",
        "MUNICIPALITY": "assessor_municipality",
    }

    for i, row in enumerate(features):
        key = str(i)
        county = county_results.get(key)
        if county:
            for src_field, dst_field in field_map.items():
                val = county.get(src_field, "")
                row[dst_field] = str(val).strip() if val is not None else ""
            row["assessor_source"] = "county"
        else:
            for dst_field in field_map.values():
                row.setdefault(dst_field, "")
            row.setdefault("assessor_source", "")

        nrhp = nrhp_matches.get(key)
        if nrhp:
            row["nrhp_listed"] = "yes"
            row["nrhp_name"] = nrhp["name"]
            row["nrhp_date_listed"] = nrhp["date_listed"]
        else:
            row["nrhp_listed"] = "no"
            row["nrhp_name"] = ""
            row["nrhp_date_listed"] = ""

    all_fields = list(existing_fields)
    for f in new_fields:
        if f not in all_fields:
            all_fields.append(f)

    with open(features_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        w.writeheader()
        for row in features:
            w.writerow(row)

    print(f"  Wrote {len(features)} rows, {len(all_fields)} columns to building_features.csv")

    # --- Summary stats ---
    print("\n=== Summary ===")
    county_total = sum(1 for v in county_results.values() if v is not None)
    city_total = sum(1 for v in county_results.values() if v is None)
    print(f"County parcel matches: {county_total}")
    print(f"No county match (city): {city_total}")
    print(f"NRHP listed: {len(nrhp_matches)}")

    yr_filled = sum(1 for i in range(n) if county_results.get(str(i)) and county_results[str(i)].get("YEARBLT"))
    sqft_filled = sum(1 for i in range(n) if county_results.get(str(i)) and county_results[str(i)].get("RESQFT"))
    print(f"Year built coverage: {yr_filled}/{n} ({100*yr_filled/n:.1f}%)")
    print(f"Sq ft coverage: {sqft_filled}/{n} ({100*sqft_filled/n:.1f}%)")


if __name__ == "__main__":
    main()
