"""
Fetch Nearmap aerial tiles for buildings in unified_buildings.csv.
Targets the 2025-05-17 capture (one day post-tornado) for damage assessment.
Also fetches a pre-event capture for before/after comparison.
"""
import csv, os, json, math, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

API_KEY = open(os.path.expanduser("~/.nearmap_apikey")).read().strip()
CSV_PATH = Path(__file__).parent / "unified_buildings.csv"
OUT_DIR = Path(__file__).parent / "images" / "nearmap"
META_PATH = Path(__file__).parent / "nearmap_meta.json"

ZOOM = 20  # ~0.15m/px at this latitude
TILE_SIZE = 256
POST_DATE = "2025-05-17"
PRE_DATE = "2025-05-16"
MAX_WORKERS = 4
BATCH_SIZE = 25


def lat_lon_to_tile(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def fetch_tile(x, y, zoom, until_date, out_path):
    url = (f"https://api.nearmap.com/tiles/v3/Vert/{zoom}/{x}/{y}.jpg"
           f"?apikey={API_KEY}&until={until_date}")
    req = urllib.request.Request(url)
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        with open(out_path, "wb") as f:
            f.write(resp.read())
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise


def fetch_building_tiles(idx, row):
    lat, lon = row["latitude"], row["longitude"]
    if not lat or not lon:
        return idx, {"status": "no_coords"}

    lat, lon = float(lat), float(lon)
    cx, cy = lat_lon_to_tile(lat, lon, ZOOM)

    bld_dir = OUT_DIR / f"bld_{idx:05d}"
    bld_dir.mkdir(parents=True, exist_ok=True)

    result = {"status": "OK", "lat": lat, "lon": lon,
              "address": row.get("complete_address", ""), "source": row["source"],
              "tile_x": cx, "tile_y": cy}

    # 3x3 grid around the center tile for context
    for label, until in [("post", POST_DATE), ("pre", PRE_DATE)]:
        tiles_ok = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                tx, ty = cx + dx, cy + dy
                path = bld_dir / f"{label}_{dx+1}_{dy+1}.jpg"
                if path.exists():
                    tiles_ok += 1
                    continue
                try:
                    if fetch_tile(tx, ty, ZOOM, until, path):
                        tiles_ok += 1
                except Exception as e:
                    result[f"{label}_error"] = str(e)
        result[f"{label}_tiles"] = tiles_ok

    return idx, result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    meta = {}
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text())

    with open(CSV_PATH) as f:
        buildings = list(csv.DictReader(f))

    todo = [(i, b) for i, b in enumerate(buildings) if str(i) not in meta]

    print(f"Total buildings: {len(buildings)}, already fetched: {len(meta)}, "
          f"remaining: {len(todo)}")

    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for batch_start in range(0, len(todo), BATCH_SIZE):
            batch = todo[batch_start:batch_start + BATCH_SIZE]
            futures = {ex.submit(fetch_building_tiles, i, b): i for i, b in batch}

            for fut in as_completed(futures):
                idx, result = fut.result()
                meta[str(idx)] = result
                done += 1
                if done % 10 == 0:
                    print(f"  {done}/{len(todo)} done")

            META_PATH.write_text(json.dumps(meta, indent=2))
            print(f"  Saved checkpoint ({batch_start + len(batch)}/{len(todo)})")
            time.sleep(0.5)

    ok = sum(1 for v in meta.values() if v.get("status") == "OK")
    print(f"\nDone. {ok}/{len(meta)} with Nearmap tiles.")


if __name__ == "__main__":
    main()
