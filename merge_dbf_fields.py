"""Merge additional fields from the city DBF parcel file into building_features.csv.

Adds: story code, occupancy code, historic conservation district,
owner-occupied status, neighborhood, ward, vacant building year.
"""
import re
import numpy as np
import pandas as pd
from dbfread import DBF

SCRATCH = "/private/tmp/claude-502/-Users-becca-Code-disasters-imageCV/17aa8ff0-86dd-49a1-9843-b102d149c4aa/scratchpad"
DBF_PATH = f"{SCRATCH}/par_dbf/par.dbf"
FEATURES_PATH = "/Users/becca/Code/disasters/imageCV/building_features.csv"

STREET_ABBREV = {
    "AVENUE": "AV", "AVE": "AV",
    "BOULEVARD": "BL", "BLVD": "BL",
    "CIRCLE": "CIR", "COURT": "CT",
    "DRIVE": "DR", "HIGHWAY": "HWY",
    "LANE": "LN", "PARKWAY": "PY", "PKWY": "PY",
    "PLACE": "PL", "ROAD": "RD", "STREET": "ST",
    "TERRACE": "TER", "WAY": "WY",
}
DIRECTION_ABBREV = {
    "NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W",
    "NORTHEAST": "NE", "NORTHWEST": "NW",
    "SOUTHEAST": "SE", "SOUTHWEST": "SW",
}


def normalize_addr(addr):
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
    print("Loading DBF...")
    db = DBF(DBF_PATH, load=True)
    dbf_df = pd.DataFrame(db.records)
    print(f"  DBF records: {len(dbf_df)}")

    dbf_df["norm_addr"] = dbf_df["SITEADDR"].apply(normalize_city_addr)
    dbf_lookup = {}
    for _, row in dbf_df.iterrows():
        key = row["norm_addr"]
        if key:
            dbf_lookup[key] = row

    print(f"  Unique normalized addresses: {len(dbf_lookup)}")

    print("Loading building features...")
    feat = pd.read_csv(FEATURES_PATH, low_memory=False)
    print(f"  Buildings: {len(feat)}")

    new_cols = {
        "assessor_story_code": np.nan,
        "assessor_occ_code": np.nan,
        "assessor_historic_district": np.nan,
        "assessor_owner_occupied": "",
        "assessor_neighborhood": np.nan,
        "assessor_ward": np.nan,
        "assessor_vacant_year": np.nan,
    }
    for col, default in new_cols.items():
        if col not in feat.columns:
            feat[col] = default

    matched = 0
    for idx, row in feat.iterrows():
        addr = row.get("complete_address", "")
        norm = normalize_addr(addr)
        if norm and norm in dbf_lookup:
            dbf_row = dbf_lookup[norm]
            feat.at[idx, "assessor_story_code"] = dbf_row.get("BDG1STRYCD", 0) or 0
            feat.at[idx, "assessor_occ_code"] = dbf_row.get("BDG1OCCCD", 0) or 0
            feat.at[idx, "assessor_historic_district"] = dbf_row.get("HSCONSERV", 0) or 0
            feat.at[idx, "assessor_owner_occupied"] = str(dbf_row.get("OWNEROCC", "")).strip()
            feat.at[idx, "assessor_neighborhood"] = dbf_row.get("NBRHD", 0) or 0
            feat.at[idx, "assessor_ward"] = dbf_row.get("WARD", 0) or 0
            vby = dbf_row.get("VACBLDGYR", 0) or 0
            feat.at[idx, "assessor_vacant_year"] = vby if vby > 0 else np.nan
            matched += 1

    print(f"  Matched by address: {matched}")

    # For city-spatial-matched buildings without address, try matching via
    # the HANDLE field through the basic info CSV.
    basic = pd.read_csv(
        f"{SCRATCH}/parcels_basic.csv", low_memory=False,
        dtype=str
    )
    basic_handle_to_addr = {}
    for _, brow in basic.iterrows():
        h = str(brow.get("HANDLE", "")).strip()
        sa = str(brow.get("SITEADDR", "")).strip()
        if h and sa:
            basic_handle_to_addr[h] = sa

    dbf_handle_lookup = {}
    for _, row in dbf_df.iterrows():
        h = str(row.get("HANDLE", "")).strip()
        if h:
            dbf_handle_lookup[h] = row

    # For buildings that matched city_spatial but didn't match above,
    # try using the assessor_owner or assessor_zoning as a clue — but
    # the most reliable link is HANDLE. The spatial join script may not
    # have stored HANDLE. Skip this path for now.

    # Coverage stats
    for col in new_cols:
        if col == "assessor_owner_occupied":
            filled = (feat[col].astype(str).str.strip() != "").sum()
        elif col == "assessor_vacant_year":
            filled = feat[col].notna().sum()
        else:
            filled = ((feat[col].notna()) & (feat[col] != 0)).sum()
        print(f"  {col}: {filled}/{len(feat)} ({100*filled/len(feat):.1f}%)")

    feat.to_csv(FEATURES_PATH, index=False)
    print(f"\nSaved {FEATURES_PATH}: {len(feat)} rows, {len(feat.columns)} columns")

    # Cross-tab story code vs CLIP-predicted stories
    print("\n=== Decoding BDG1STRYCD: cross-tab vs predicted_stories ===")
    has_both = feat[
        (feat.assessor_story_code > 0) &
        feat.predicted_stories.notna()
    ].copy()
    has_both["pred_s"] = pd.to_numeric(has_both["predicted_stories"], errors="coerce")
    if len(has_both) > 0:
        ct = pd.crosstab(
            has_both["assessor_story_code"],
            has_both["pred_s"],
        )
        ct.index.name = "story_code"
        ct.columns.name = "CLIP_stories"
        print(ct.to_string())
    else:
        print("  No overlap found")

    # Also check vs undergrad number_stories from unified
    try:
        uni = pd.read_csv(
            "/Users/becca/Code/disasters/imageCV/unified_buildings.csv",
            low_memory=False
        )
        feat_ug = feat.copy()
        feat_ug["ug_stories"] = uni["number_stories"]
        has_ug = feat_ug[
            (feat_ug.assessor_story_code > 0) &
            (feat_ug.ug_stories.notna()) &
            (feat_ug.ug_stories > 0)
        ]
        if len(has_ug) > 0:
            print("\n=== Story code vs undergrad-surveyed stories ===")
            ct2 = pd.crosstab(
                has_ug["assessor_story_code"],
                has_ug["ug_stories"],
            )
            ct2.index.name = "story_code"
            ct2.columns.name = "undergrad_stories"
            print(ct2.to_string())
    except Exception as e:
        print(f"  Could not cross-check undergrad stories: {e}")


if __name__ == "__main__":
    main()
