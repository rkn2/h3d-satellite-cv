"""
Fetch Google Street View images for all buildings in unified_buildings.csv.
Pre-event imagery for archetype classification (stories, roof shape, wall material).
"""
import csv, os, json, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

API_KEY = "AIzaSyD-NTIH1LDvWjdg5ZxuvlyNNpnrJrdV2Ro"
CSV_PATH = Path(__file__).parent / "unified_buildings.csv"
OUT_DIR = Path(__file__).parent / "images" / "streetview"
META_PATH = Path(__file__).parent / "streetview_meta.json"

HEADINGS = [0, 90, 180, 270]
IMG_SIZE = "640x480"
BATCH_SIZE = 50
MAX_WORKERS = 8


def fetch_metadata(lat, lon):
    url = (f"https://maps.googleapis.com/maps/api/streetview/metadata"
           f"?location={lat},{lon}&key={API_KEY}")
    resp = urllib.request.urlopen(url, timeout=15)
    return json.loads(resp.read())


def fetch_image(lat, lon, heading, out_path):
    url = (f"https://maps.googleapis.com/maps/api/streetview"
           f"?size={IMG_SIZE}&location={lat},{lon}"
           f"&heading={heading}&pitch=10&fov=90&key={API_KEY}")
    urllib.request.urlretrieve(url, out_path)


def process_building(idx, row):
    lat, lon = row["latitude"], row["longitude"]
    if not lat or not lon:
        return idx, {"status": "no_coords"}

    bld_dir = OUT_DIR / f"bld_{idx:05d}"

    meta = fetch_metadata(lat, lon)
    if meta.get("status") != "OK":
        return idx, {"status": meta.get("status", "UNKNOWN"), "pano_date": None}

    pano_date = meta.get("date", "")
    result = {"status": "OK", "pano_date": pano_date, "lat": lat, "lon": lon,
              "address": row.get("complete_address", ""), "source": row["source"]}

    bld_dir.mkdir(parents=True, exist_ok=True)
    for heading in HEADINGS:
        img_path = bld_dir / f"h{heading}.jpg"
        if not img_path.exists():
            try:
                fetch_image(lat, lon, heading, img_path)
            except Exception as e:
                result[f"h{heading}_error"] = str(e)

    return idx, result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    meta = {}
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text())

    with open(CSV_PATH) as f:
        buildings = list(csv.DictReader(f))

    todo = [(i, b) for i, b in enumerate(buildings)
            if str(i) not in meta or meta[str(i)].get("status") == "no_coords"]

    print(f"Total buildings: {len(buildings)}, already fetched: {len(meta)}, "
          f"remaining: {len(todo)}")

    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for batch_start in range(0, len(todo), BATCH_SIZE):
            batch = todo[batch_start:batch_start + BATCH_SIZE]
            futures = {ex.submit(process_building, i, b): i for i, b in batch}

            for fut in as_completed(futures):
                idx, result = fut.result()
                meta[str(idx)] = result
                done += 1
                if done % 25 == 0:
                    print(f"  {done}/{len(todo)} done")

            META_PATH.write_text(json.dumps(meta, indent=2))
            print(f"  Saved checkpoint ({batch_start + len(batch)}/{len(todo)})")

    ok = sum(1 for v in meta.values() if v.get("status") == "OK")
    print(f"\nDone. {ok}/{len(meta)} with Street View imagery.")


if __name__ == "__main__":
    main()
