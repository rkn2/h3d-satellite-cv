"""Zero-shot CLIP feature extraction + distance-to-track.

Extracts 12 additional building features from cached CLIP embeddings
using zero-shot text-prompt classification, and computes distance to
the May 16 2025 St. Louis tornado track.

Run: uv run --with pandas,numpy,torch,open-clip-torch,pillow,shapely \
         --python 3.12 python3 extract_features.py
"""
import csv, json, math
from pathlib import Path

import numpy as np
import torch
import open_clip

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BASE = Path(__file__).parent
CSV_PATH = BASE / "unified_buildings.csv"
SV_EMBED = BASE / "sv_embeddings.npz"
NM_PRE_EMBED = BASE / "nm_cropped_embeddings.npz"
NM_POST_EMBED = BASE / "post_embeddings.npz"
DAT_GEOJSON = Path("/private/tmp/claude-502/-Users-becca-Code-disasters-imageCV/"
                   "17aa8ff0-86dd-49a1-9843-b102d149c4aa/scratchpad/dat_polygons.geojson")
OUT_PATH = BASE / "building_features.csv"

FEATURES_SV = {
    "shared_wall": {
        "classes": ["yes", "no"],
        "prompts": [
            "a building with a blank brick party wall on the side with no windows on the side wall, attached to an adjacent building",
            "a freestanding detached building with windows visible on the side walls, standing alone",
        ],
    },
    "fenestration_level": {
        "classes": ["low", "medium", "high"],
        "prompts": [
            "a building facade with very few small windows, mostly solid brick or stone wall",
            "a building facade with moderate sized windows, roughly equal amounts of wall and glass",
            "a building facade with large display windows or a glass storefront, mostly glass",
        ],
    },
    "parapet_present": {
        "classes": ["yes", "no"],
        "prompts": [
            "a building with a flat parapet wall extending above the roofline hiding the roof",
            "a building with a pitched roof with visible eaves, gable ends, and no parapet wall",
        ],
    },
    "soffit_present": {
        "classes": ["yes", "no"],
        "prompts": [
            "a building with visible soffit panels and roof eaves overhang extending past the walls",
            "a building with no eaves overhang, either a flat roof or walls going straight to the roof edge",
        ],
    },
    "porch_present": {
        "classes": ["yes", "no"],
        "prompts": [
            "a residential house with a front porch, covered porch, or veranda with columns or posts",
            "a building entrance directly from the sidewalk with no porch or covered entry",
        ],
    },
    "garage_present": {
        "classes": ["yes", "no"],
        "prompts": [
            "a building with a garage door, carport, or vehicle entrance visible on the facade",
            "a building with no garage door or vehicle entrance, only pedestrian doors",
        ],
    },
    "foundation_type": {
        "classes": ["raised", "slab", "unclear"],
        "prompts": [
            "a building elevated on a raised foundation with visible crawl space, basement windows, or front steps leading up to the entrance",
            "a building sitting directly on a concrete slab at ground level with the entrance at sidewalk height",
            "a building where the foundation type cannot be determined from the photograph",
        ],
    },
    "wall_cladding_detail": {
        "classes": ["brick_only", "brick_vinyl_mix", "vinyl_siding", "stone", "stucco", "wood_siding"],
        "prompts": [
            "a building with solid red or brown brick walls, all brick exterior, no other cladding",
            "a building with a mix of brick on the lower portion and vinyl or aluminum siding on upper floors",
            "a building covered entirely in vinyl or aluminum horizontal lap siding",
            "a building with natural stone walls, limestone, or rough-cut stone exterior",
            "a building with smooth stucco or cement plaster exterior walls",
            "a building with horizontal wood clapboard siding or wood shingle siding",
        ],
    },
}

FEATURES_NM_PRE = {
    "roof_cover_type": {
        "classes": ["asphalt_shingles", "metal", "flat_membrane", "tile"],
        "prompts": [
            "aerial view of a roof covered with dark asphalt shingles in a regular pattern",
            "aerial view of a shiny metal roof with standing seam panels or corrugated metal",
            "aerial view of a flat roof with light-colored membrane, gravel, or built-up roofing",
            "aerial view of a roof with clay or concrete roof tiles in a curved or flat pattern",
        ],
    },
    "roof_condition": {
        "classes": ["good", "fair", "poor"],
        "prompts": [
            "aerial view of a well-maintained roof with uniform color and intact material, no patches or wear",
            "aerial view of a roof showing some wear with minor discoloration or aging but still functional",
            "aerial view of a deteriorated roof with missing shingles, visible patches, moss, or worn-through areas",
        ],
    },
}

FEATURES_NM_POST = {
    "roof_damage_visible": {
        "classes": ["none", "partial", "major", "total_loss"],
        "prompts": [
            "aerial view of an intact roof after a storm with no visible damage, clean and complete",
            "aerial view of a roof with partial damage, some missing shingles or small holes but mostly intact",
            "aerial view of a roof with major damage, large sections missing, exposed structure visible",
            "aerial view of a building with total roof loss, no roof remaining, interior or ground visible from above",
        ],
    },
    "debris_present": {
        "classes": ["yes", "no"],
        "prompts": [
            "aerial view of a roof or property covered with storm debris, fallen tree branches, and scattered material",
            "aerial view of a clean roof and property with no visible debris or fallen material",
        ],
    },
}


def load_embeddings(path):
    data = np.load(path, allow_pickle=True)
    return data["embeddings"].item()


def encode_prompts(model, tokenizer, prompts, device):
    tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()


def classify_zero_shot(embeddings, text_features, classes, temperature=100.0):
    results = {}
    text_mat = text_features.astype(np.float32)
    for key, emb in embeddings.items():
        emb_norm = emb / (np.linalg.norm(emb) + 1e-10)
        sims = emb_norm @ text_mat.T
        scaled = sims * temperature
        exp_s = np.exp(scaled - scaled.max())
        probs = exp_s / exp_s.sum()
        best = int(np.argmax(probs))
        results[key] = (classes[best], float(probs[best]))
    return results


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def compute_distances(buildings, geojson_path):
    if not geojson_path.exists():
        print(f"  DAT GeoJSON not found at {geojson_path}, skipping distance")
        return {}

    with open(geojson_path) as f:
        dat = json.load(f)

    from shapely.geometry import shape
    ef3_polys = []
    for feat in dat["features"]:
        ts = feat["properties"]["stormdate"]
        ef = feat["properties"]["efscale"]
        if 1747400000000 < ts < 1747430000000 and ef == "EF3":
            geom = shape(feat["geometry"])
            ef3_polys.append(geom)

    if not ef3_polys:
        print("  No EF3 polygons found for May 16 2025")
        return {}

    centroids = [(p.centroid.y, p.centroid.x) for p in ef3_polys]
    print(f"  Found {len(ef3_polys)} EF3 polygons, centroids: {[(f'{c[0]:.4f}', f'{c[1]:.4f}') for c in centroids]}")

    distances = {}
    for i, bld in enumerate(buildings):
        lat = float(bld["latitude"])
        lon = float(bld["longitude"])
        min_d = min(haversine_km(lat, lon, clat, clon) for clat, clon in centroids)
        distances[str(i)] = round(min_d, 4)
    return distances


def main():
    print("Loading CLIP model (ViT-B-32 to match 512-dim cached embeddings)...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=DEVICE
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    with open(CSV_PATH) as f:
        buildings = list(csv.DictReader(f))
    n = len(buildings)
    print(f"Loaded {n} buildings.")

    print("Loading cached embeddings...")
    sv_embeds = load_embeddings(SV_EMBED)
    print(f"  Street View: {len(sv_embeds)} buildings")

    nm_pre_embeds = load_embeddings(NM_PRE_EMBED) if NM_PRE_EMBED.exists() else {}
    print(f"  Nearmap pre: {len(nm_pre_embeds)} buildings")

    nm_post_embeds = load_embeddings(NM_POST_EMBED) if NM_POST_EMBED.exists() else {}
    print(f"  Nearmap post: {len(nm_post_embeds)} buildings")

    all_features = {}
    for i in range(n):
        all_features[str(i)] = {}

    def run_feature_set(feature_defs, embeddings, source_name):
        for feat_name, spec in feature_defs.items():
            print(f"  Classifying {feat_name} from {source_name}...")
            text_feats = encode_prompts(model, tokenizer, spec["prompts"], DEVICE)
            results = classify_zero_shot(embeddings, text_feats, spec["classes"])
            count = 0
            for key, (cls, conf) in results.items():
                if key in all_features:
                    all_features[key][feat_name] = cls
                    all_features[key][f"{feat_name}_conf"] = round(conf, 3)
                    count += 1
            print(f"    → {count} buildings classified")

    print("\n--- Street View features ---")
    run_feature_set(FEATURES_SV, sv_embeds, "Street View")

    print("\n--- Nearmap pre-event features ---")
    run_feature_set(FEATURES_NM_PRE, nm_pre_embeds, "Nearmap pre")

    print("\n--- Nearmap post-event features ---")
    run_feature_set(FEATURES_NM_POST, nm_post_embeds, "Nearmap post")

    print("\n--- Computing distance to EF3 track ---")
    distances = compute_distances(buildings, DAT_GEOJSON)
    for key, d in distances.items():
        if key in all_features:
            all_features[key]["distance_km"] = d
    print(f"  → {len(distances)} distances computed")

    print("\n--- Writing output ---")
    base_cols = list(buildings[0].keys())
    feat_cols = set()
    for feats in all_features.values():
        feat_cols.update(feats.keys())
    feat_cols = sorted(feat_cols)

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base_cols + feat_cols)
        writer.writeheader()
        for i, bld in enumerate(buildings):
            row = dict(bld)
            row.update(all_features.get(str(i), {}))
            writer.writerow(row)

    print(f"\nWrote {n} rows × {len(base_cols) + len(feat_cols)} columns to {OUT_PATH}")

    print("\n--- Summary ---")
    for col in feat_cols:
        filled = sum(1 for k in all_features if col in all_features[k])
        if not col.endswith("_conf"):
            vals = {}
            for k in all_features:
                v = all_features[k].get(col)
                if v is not None:
                    vals[v] = vals.get(v, 0) + 1
            dist_str = ", ".join(f"{k}:{v}" for k, v in sorted(vals.items(), key=lambda x: -x[1])[:5])
            print(f"  {col}: {filled}/{n} filled — {dist_str}")

    print("\n--- Sample (first 5 buildings with SV) ---")
    sample_keys = [k for k in sorted(all_features.keys(), key=int) if all_features[k]][:5]
    for k in sample_keys:
        feats = all_features[k]
        non_conf = {kk: vv for kk, vv in feats.items() if not kk.endswith("_conf")}
        print(f"  bld_{int(k):05d}: {non_conf}")


if __name__ == "__main__":
    main()
