"""Add heritage, ownership, and timeline features to building_features.csv.

Matches the 72 heritage/ownership columns from the IJAH quad-state/Nashville paper.
Sources: NRHP database (NPS ArcGIS), assessor data, occupancy data.
"""
import json
import re
import pandas as pd
import numpy as np
from pathlib import Path

FEAT_PATH = Path(__file__).parent / "building_features.csv"
UNI_PATH = Path(__file__).parent / "unified_buildings.csv"
SCRATCH = Path("/private/tmp/claude-502/-Users-becca-Code-disasters-imageCV/"
               "17aa8ff0-86dd-49a1-9843-b102d149c4aa/scratchpad")

def load_nrhp():
    points_path = SCRATCH / "nrhp_stl.geojson"
    districts_path = SCRATCH / "nrhp_districts_stl.geojson"

    nrhp_points = []
    if points_path.exists():
        with open(points_path) as f:
            data = json.load(f)
        for feat in data.get("features", []):
            p = feat["properties"]
            g = feat.get("geometry")
            if g and g.get("coordinates"):
                nrhp_points.append({
                    "name": p.get("RESNAME", ""),
                    "address": p.get("Address", ""),
                    "city": p.get("City", ""),
                    "restype": p.get("ResType", ""),
                    "cert_date": p.get("CertDate", ""),
                    "lon": g["coordinates"][0],
                    "lat": g["coordinates"][1],
                })

    nrhp_districts = []
    if districts_path.exists():
        with open(districts_path) as f:
            data = json.load(f)
        for feat in data.get("features", []):
            p = feat["properties"]
            if p.get("ResType") == "district":
                from shapely.geometry import shape
                try:
                    geom = shape(feat["geometry"])
                    if geom.is_valid:
                        nrhp_districts.append({
                            "name": p.get("RESNAME", ""),
                            "geom": geom,
                            "cert_date": p.get("CertDate", ""),
                        })
                except Exception:
                    pass

    return nrhp_points, nrhp_districts


def parse_cert_year(cert_date):
    if not cert_date:
        return None
    m = re.search(r"(\d{2})$", str(cert_date))
    if m:
        yr = int(m.group(1))
        return 1900 + yr if yr > 25 else 2000 + yr
    return None


def classify_owner(name):
    if not isinstance(name, str) or not name.strip():
        return "unknown"
    name = name.upper().strip()

    gov_kw = ["CITY OF", "STATE OF", "HOUSING AUTH", "UNITED STATES", "COUNTY",
              "LAND REUTILIZATION", "PUBLIC SCHOOL", "BOARD OF EDUCATION",
              "POLICE", "FIRE DEPT", "GOVERNMENT", "MUNICIPAL"]
    if any(k in name for k in gov_kw):
        return "government"

    rel_kw = ["CHURCH", "TEMPLE", "MOSQUE", "SYNAGOGUE", "CONGREGATION",
              "PARISH", "MINISTRY", "BAPTIST", "METHODIST", "CATHOLIC",
              "LUTHERAN", "PENTECOSTAL", "AME", "CHRISTIAN", "EPISCOPAL",
              "PRESBYTERIAN", "ADVENTIST", "APOSTOLIC", "ZION", "BETHEL",
              "GOSPEL", "SALVATION ARMY"]
    if any(k in name for k in rel_kw):
        return "religious"

    biz_kw = ["LLC", "INC", "CORP", "LP", "LTD", "COMPANY", "CO.", "ASSOC",
              "PARTNERS", "ENTERPRISES", "DEVELOPMENT", "PROPERTIES", "REALTY",
              "MANAGEMENT", "HOLDINGS", "INVESTMENTS", "GROUP", "FUND",
              "FOUNDATION", "SOCIETY", "TRUST"]
    if any(k in name for k in biz_kw):
        return "business"

    ngo_kw = ["HABITAT FOR HUMANITY", "NEIGHBORHOOD", "COMMUNITY DEV",
              "HISTORIC", "PRESERVATION", "NON-PROFIT", "NONPROFIT"]
    if any(k in name for k in ngo_kw):
        return "ngo"

    return "individual"


def main():
    print("Loading data...")
    feat = pd.read_csv(FEAT_PATH)
    uni = pd.read_csv(UNI_PATH)
    n = len(feat)

    # Ensure we have columns from unified
    if "construction_type_u" not in feat.columns:
        feat["construction_type_u"] = uni["construction_type_u"]
    if "occupany_u" not in feat.columns:
        feat["occupany_u"] = uni["occupany_u"]
    if "wall_cladding_u" not in feat.columns:
        feat["wall_cladding_u"] = uni["wall_cladding_u"]

    # ── 1. NRHP matching ──
    print("Loading NRHP data...")
    try:
        from shapely.geometry import Point
        nrhp_points, nrhp_districts = load_nrhp()
        has_shapely = True
    except ImportError:
        nrhp_points, nrhp_districts = load_nrhp.__wrapped__() if hasattr(load_nrhp, '__wrapped__') else ([], [])
        has_shapely = False

    print(f"  NRHP points: {len(nrhp_points)}, districts: {len(nrhp_districts)}")

    feat["national_register_listing_year"] = np.nan
    feat["building_name_listing"] = ""
    feat["building_name_current"] = ""
    feat["located_in_historic_district"] = "no"

    nrhp_match_count = 0
    district_match_count = 0

    # Match buildings to NRHP points by proximity (< 50m ≈ 0.00045 degrees)
    for i in range(n):
        lat_i = feat.loc[i, "latitude"]
        lon_i = feat.loc[i, "longitude"]
        if pd.isna(lat_i) or pd.isna(lon_i):
            continue

        best_dist = 0.00045  # ~50m threshold
        best_match = None
        for pt in nrhp_points:
            d = ((lat_i - pt["lat"])**2 + (lon_i - pt["lon"])**2)**0.5
            if d < best_dist:
                best_dist = d
                best_match = pt

        if best_match:
            nrhp_match_count += 1
            feat.loc[i, "building_name_listing"] = best_match["name"]
            yr = parse_cert_year(best_match["cert_date"])
            if yr:
                feat.loc[i, "national_register_listing_year"] = yr

        # Check if inside any NRHP district polygon
        if has_shapely and nrhp_districts:
            pt = Point(lon_i, lat_i)
            for dist in nrhp_districts:
                if dist["geom"].contains(pt):
                    feat.loc[i, "located_in_historic_district"] = "yes"
                    district_match_count += 1
                    break

    print(f"  NRHP point matches (< 50m): {nrhp_match_count}")
    print(f"  In historic district: {district_match_count}")

    # ── 2. Owner classification ──
    print("Classifying owners...")
    owner_col = feat.get("assessor_owner", pd.Series(dtype=str))
    owner_class = owner_col.apply(classify_owner)

    feat["owner_individual"] = (owner_class == "individual").map({True: "yes", False: "no"})
    feat["owner_business"] = (owner_class == "business").map({True: "yes", False: "no"})
    feat["owner_government"] = (owner_class == "government").map({True: "yes", False: "no"})
    feat["owner_ngo"] = (owner_class == "ngo").map({True: "yes", False: "no"})
    feat["owner_religious "] = (owner_class == "religious").map({True: "yes", False: "no"})
    feat["owner_unknown"] = (owner_class == "unknown").map({True: "yes", False: "no"})

    print(f"  Owner distribution: {owner_class.value_counts().to_dict()}")

    # ── 3. Unit classification ──
    units = feat.get("assessor_units", pd.Series(dtype=float))
    feat["single_unit"] = "no"
    feat["multiple_unit"] = "no"
    feat.loc[units == 1, "single_unit"] = "yes"
    feat.loc[units > 1, "multiple_unit"] = "yes"
    feat.loc[units.isna() | (units == 0), "single_unit"] = "un"
    feat.loc[units.isna() | (units == 0), "multiple_unit"] = "un"

    # ── 4. Construction material flags ──
    print("Setting construction material flags...")
    wall_code = feat.get("assessor_wall_code", pd.Series(dtype=float))
    const_type = feat.get("construction_type_u", pd.Series(dtype=str)).fillna("")
    cladding = feat.get("wall_cladding_u", pd.Series(dtype=str)).fillna("")

    is_brick = (wall_code == 31) | (wall_code == 42) | const_type.str.contains("masonry", case=False, na=False)
    is_stone = (wall_code == 42)  # tentative
    is_wood = (wall_code == 32) | const_type.str.contains("wood", case=False, na=False)

    for prefix in ["h", "v"]:
        feat[f"const_material_{prefix}_brick"] = is_brick.map({True: "yes", False: "no"})
        feat[f"const_material_{prefix}_stone"] = is_stone.map({True: "yes", False: "no"})
        feat[f"const_material_{prefix}_wood"] = is_wood.map({True: "yes", False: "no"})
        feat[f"const_material_{prefix}_mud"] = "no"
        feat[f"const_material_{prefix}_rf_masonry"] = "no"
        feat[f"const_material_{prefix}_rglr_stone"] = "no"
        feat[f"const_material_{prefix}_rf_conc"] = "no"
        feat[f"const_material_{prefix}_ir_stone"] = "no"
        feat[f"const_material_{prefix}_othr"] = (~is_brick & ~is_stone & ~is_wood).map({True: "yes", False: "no"})

    # ── 5. Property type flags ──
    print("Setting property type flags...")
    occ = feat.get("occupany_u", pd.Series(dtype=str)).fillna("").str.lower()

    prop_types = {
        "prop_residential facility": occ.str.contains("residential", na=False),
        "prop_commercial / exchange facility": occ.str.contains("commercial|business|mercantile", na=False),
        "prop_religious": (owner_class == "religious") | occ.str.contains("religious", na=False),
        "prop_educational facility": occ.str.contains("school|educational", na=False),
        "prop_culture _entertainment_facility": occ.str.contains("museum|entertainment|assembly", na=False),
        "prop_industrial_facility": occ.str.contains("factory|industrial", na=False),
        "prop_health / welfare facility": occ.str.contains("hospital|health", na=False),
        "prop_law / government facility": (owner_class == "government"),
    }

    all_prop_cols = [
        "prop_agricultural ", "prop_archaeological", "prop_battlefield",
        "prop_cave", "prop_commemorative structure or landmark",
        "prop_commercial / exchange facility", "prop_culture _entertainment_facility",
        "prop_ecosystem", "prop_educational facility", "prop_forest",
        "prop_habitat", "prop_health / welfare facility",
        "prop_industrial_facility", "prop_infrastructure", "prop_island(s)",
        "prop_lake", "prop_law / government facility", "prop_marine zone",
        "prop_military", "prop_mine", "prop_mountain", "prop_nature",
        "prop_park / garden", "prop_parking / storage facility",
        "prop_religious", "prop_residential facility",
        "prop_river catchment system", "prop_rock formation",
        "prop_scenic area", "prop_sea", "prop_sports_facility",
        "prop_transportation facility", "prop_underground facility",
        "prop_unilities_facility", "prop_volcano", "prop_zoological park",
    ]

    for col in all_prop_cols:
        if col in prop_types:
            feat[col] = prop_types[col].map({True: "yes", False: "no"})
        else:
            feat[col] = "no"

    # ── 6. Heritage value flags ──
    is_nrhp = feat["national_register_listing_year"].notna() | (feat["located_in_historic_district"] == "yes")
    feat["prop_val_evidential"] = "no value"
    feat["prop_val_historical"] = is_nrhp.map({True: "considerable", False: "no value"})
    feat["prop_val_aesthetic"] = is_nrhp.map({True: "limited", False: "no value"})
    feat["prop_val_communal"] = "no value"

    # ── 7. Heritage designation flags ──
    feat["world_heritage_property"] = "no"
    feat["hague_convention"] = "no"
    feat["sub_national_heritage _list"] = "no"
    feat["iucn_protected_area"] = "no"
    feat["property_of_local_significance"] = (feat["located_in_historic_district"] == "yes").map({True: "yes", False: "no"})

    # ── 8. Timeline columns ──
    print("Setting timeline columns...")
    yr_built = feat.get("assessor_year_built", pd.Series(dtype=float))

    feat["existed_during_tornado"] = "yes"
    feat["building_existed_during_tornado"] = "yes"
    feat["building_in_use_during_tornado"] = "yes"
    feat["building_use_during_tornado"] = feat.get("occupany_u", "unknown")

    feat["buidling_existed_5_yrs_before_tornado"] = (yr_built <= 2020).map({True: "yes", False: "no"})
    feat.loc[yr_built.isna() | (yr_built == 0), "buidling_existed_5_yrs_before_tornado"] = "un"

    feat["buidling_existed_3_yrs_before_tornado"] = (yr_built <= 2022).map({True: "yes", False: "no"})
    feat.loc[yr_built.isna() | (yr_built == 0), "buidling_existed_3_yrs_before_tornado"] = "un"

    feat["buidling_existed_1_yrs_before_tornado"] = (yr_built <= 2024).map({True: "yes", False: "no"})
    feat.loc[yr_built.isna() | (yr_built == 0), "buidling_existed_1_yrs_before_tornado"] = "un"

    feat["buidling_use_before_tornado"] = feat.get("occupany_u", "unknown")
    feat["buidling_use_after_tornado"] = "unknown"
    feat["buidling_use_plan_after_tornado"] = "unknown"

    feat["building_demolished_1_yrs_after_tornado"] = "un"
    feat["building_demolished_3_yrs_after_tornado"] = "un"
    feat["building_demolished_5_yrs_after_tornado"] = "un"

    # ── 9. Retrofit columns ──
    feat["retrofit_present_u"] = "un"
    feat["retrofit_type_u"] = "not_found"
    feat["retrofit_type_unc_u"] = 2
    feat["retrofit_year_u"] = "un"

    # ── 10. Other missing columns ──
    feat["building_low_rise"] = "yes"
    feat["archetype_unc"] = 2
    feat["risk_category_16"] = 2

    # ── Save ──
    feat.to_csv(FEAT_PATH, index=False)
    new_col_count = len(feat.columns)
    print(f"\nSaved building_features.csv with {new_col_count} columns")
    print(f"  NRHP point matches: {nrhp_match_count}")
    print(f"  In historic district: {district_match_count}")
    print(f"  Owner types: {owner_class.value_counts().to_dict()}")

    # Count heritage-specific new columns
    heritage_cols = [c for c in feat.columns if any(x in c for x in
        ["const_material", "prop_", "owner_", "heritage", "hague", "iucn",
         "single_unit", "multiple_unit", "national_register", "located_in_historic",
         "building_name", "existed", "demolished", "use_before", "use_after",
         "use_plan", "retrofit", "building_low_rise", "archetype_unc",
         "risk_category", "world_heritage", "sub_national", "property_of_local",
         "building_existed", "building_in_use", "building_use_during"])]
    print(f"  Heritage/ownership/timeline columns added: {len(heritage_cols)}")


if __name__ == "__main__":
    main()
