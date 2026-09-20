"""
Classify buildings using CLIP features + trained classifiers.

Phase 1: Extract CLIP embeddings from Street View images for all buildings.
Phase 2: Train classifiers on the 379 labeled undergrad buildings.
Phase 3: Predict for unlabeled buildings (remaining undergrad + all StEER).
Phase 4: Derive Nofal archetype from predicted components.

For damage: train on StEER's 4,454 labeled buildings using Nearmap aerials.

Run fetch_streetview.py and/or fetch_nearmap.py first.
"""
import csv, json, os, pickle
from pathlib import Path

import numpy as np
import torch
import open_clip
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

CSV_PATH = Path(__file__).parent / "unified_buildings.csv"
SV_DIR = Path(__file__).parent / "images" / "streetview"
NM_DIR = Path(__file__).parent / "images" / "nearmap"
EMBED_PATH = Path(__file__).parent / "sv_embeddings.npz"
NM_EMBED_PATH = Path(__file__).parent / "nm_embeddings.npz"
MODELS_DIR = Path(__file__).parent / "models"
RESULTS_PATH = Path(__file__).parent / "classification_results.json"
OUT_PATH = Path(__file__).parent / "classified_buildings.csv"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

NOFAL_TABLE = {
    6:  {"stories": 1, "walls": "brick",               "roof": "gable",   "size": "large"},
    8:  {"stories": 3, "walls": "brick",               "roof": "gable",   "size": "large"},
    9:  {"stories": 1, "walls": "brick",               "roof": "flat",    "size": "very_large"},
    11: {"stories": 1, "walls": "brick",               "roof": "gable",   "size": "small"},
    12: {"stories": 4, "walls": "glass_curtainwall",   "roof": "flat",    "size": "very_large"},
    13: {"stories": 1, "walls": "brick",               "roof": "flat",    "size": "large"},
    14: {"stories": 2, "walls": "brick",               "roof": "flat",    "size": "large"},
    17: {"stories": 1, "walls": "wood_siding",         "roof": "gable",   "size": "small"},
    18: {"stories": 1, "walls": "brick",               "roof": "flat",    "size": "large"},
    19: {"stories": 2, "walls": "brick",               "roof": "gable",   "size": "medium"},
}


def load_clip():
    print(f"Loading CLIP ViT-L-14 (device: {DEVICE})...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="laion2b_s32b_b82k", device=DEVICE
    )
    model.eval()
    print("Loaded.")
    return model, preprocess


def extract_embedding(model, preprocess, img_dir, pattern="h*.jpg"):
    imgs = []
    for p in sorted(img_dir.glob(pattern)):
        if p.stat().st_size > 1000:
            try:
                imgs.append(preprocess(Image.open(p).convert("RGB")))
            except Exception:
                pass

    if not imgs:
        return None

    batch = torch.stack(imgs).to(DEVICE)
    with torch.no_grad():
        features = model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)
        embedding = features.mean(dim=0).cpu().numpy()

    return embedding


def extract_all_embeddings(model, preprocess, img_base, n_buildings, pattern, cache_path):
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        embeddings = dict(data["embeddings"].item())
        print(f"Loaded {len(embeddings)} cached embeddings from {cache_path.name}")
    else:
        embeddings = {}

    new_count = 0
    for i in range(n_buildings):
        if str(i) in embeddings:
            continue
        bld_dir = img_base / f"bld_{i:05d}"
        if not bld_dir.exists():
            continue
        emb = extract_embedding(model, preprocess, bld_dir, pattern)
        if emb is not None:
            embeddings[str(i)] = emb
            new_count += 1
            if new_count % 50 == 0:
                print(f"  Extracted {new_count} new embeddings...")

    if new_count > 0:
        np.savez_compressed(cache_path, embeddings=embeddings)
        print(f"  Saved {len(embeddings)} total embeddings to {cache_path.name}")

    return embeddings


def train_classifier(X, y, name):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
    )

    if len(le.classes_) > 1 and len(y) >= 10:
        n_splits = min(5, min(np.bincount(y_enc)))
        n_splits = max(2, n_splits)
        scores = cross_val_score(clf, X, y_enc, cv=n_splits, scoring="accuracy")
        print(f"  {name}: CV accuracy = {scores.mean():.3f} +/- {scores.std():.3f} "
              f"(classes: {list(le.classes_)})")
    else:
        print(f"  {name}: too few samples for CV, training on all data")

    clf.fit(X, y_enc)
    return clf, le


def derive_archetype(stories, walls, roof):
    best_match = None
    best_score = -1

    for tid, spec in NOFAL_TABLE.items():
        score = 0
        if stories == spec["stories"]:
            score += 3
        elif abs(stories - spec["stories"]) == 1:
            score += 1

        if walls == spec["walls"]:
            score += 3
        elif "brick" in str(walls) and "brick" in spec["walls"]:
            score += 2

        spec_roof = spec.get("roof", "")
        if roof == spec_roof:
            score += 2
        elif "gable" in str(roof) and "gable" in spec_roof:
            score += 1

        if score > best_score:
            best_score = score
            best_match = tid

    return best_match, best_score


def main():
    with open(CSV_PATH) as f:
        buildings = list(csv.DictReader(f))
    n = len(buildings)
    print(f"Loaded {n} buildings from unified CSV.")

    model, preprocess = load_clip()

    # --- Phase 1: Extract embeddings ---
    print("\n--- Phase 1: Extracting Street View embeddings ---")
    sv_embeds = extract_all_embeddings(
        model, preprocess, SV_DIR, n, "h*.jpg", EMBED_PATH
    )

    nm_embeds = {}
    if NM_DIR.exists():
        print("\n--- Phase 1b: Extracting Nearmap embeddings ---")
        nm_embeds = extract_all_embeddings(
            model, preprocess, NM_DIR, n, "post_*.jpg", NM_EMBED_PATH
        )

    # --- Phase 2: Train classifiers ---
    print("\n--- Phase 2: Training classifiers ---")
    MODELS_DIR.mkdir(exist_ok=True)

    # Archetype component classifiers (from undergrad labeled data)
    targets = {
        "stories": "number_stories",
        "roof": "roof_shape_u",
        "walls": "construction_type_u",
    }

    classifiers = {}
    for target_name, csv_col in targets.items():
        X_train, y_train = [], []
        for i, bld in enumerate(buildings):
            key = str(i)
            label = bld.get(csv_col, "").strip()
            if not label or key not in sv_embeds:
                continue
            # Normalize some labels
            label = label.lower().replace(" ", "")
            if target_name == "stories":
                try:
                    label = str(int(float(label)))
                except ValueError:
                    continue
            X_train.append(sv_embeds[key])
            y_train.append(label)

        if len(X_train) < 10:
            print(f"  {target_name}: only {len(X_train)} labeled samples, skipping")
            continue

        X_train = np.array(X_train)
        clf, le = train_classifier(X_train, y_train, target_name)
        classifiers[target_name] = (clf, le)
        pickle.dump((clf, le), open(MODELS_DIR / f"{target_name}_clf.pkl", "wb"))

    # Damage classifier (from StEER labeled data + Nearmap embeddings)
    if nm_embeds:
        X_dmg, y_dmg = [], []
        for i, bld in enumerate(buildings):
            key = str(i)
            label = bld.get("steer_damage_rating", "").strip()
            if not label or key not in nm_embeds:
                continue
            X_dmg.append(nm_embeds[key])
            y_dmg.append(label)

        if len(X_dmg) >= 10:
            X_dmg = np.array(X_dmg)
            clf, le = train_classifier(X_dmg, y_dmg, "damage")
            classifiers["damage"] = (clf, le)
            pickle.dump((clf, le), open(MODELS_DIR / f"damage_clf.pkl", "wb"))

    # Damage from Street View (post-event panos where available)
    X_svdmg, y_svdmg = [], []
    for i, bld in enumerate(buildings):
        key = str(i)
        label = bld.get("steer_damage_rating", "").strip()
        if not label or key not in sv_embeds:
            continue
        X_svdmg.append(sv_embeds[key])
        y_svdmg.append(label)

    if len(X_svdmg) >= 10:
        X_svdmg = np.array(X_svdmg)
        clf, le = train_classifier(X_svdmg, y_svdmg, "damage_sv")
        classifiers["damage_sv"] = (clf, le)

    # --- Phase 3: Predict ---
    print("\n--- Phase 3: Predicting ---")
    results = {}

    for i, bld in enumerate(buildings):
        key = str(i)
        result = {
            "idx": i, "source": bld["source"],
            "lat": bld["latitude"], "lon": bld["longitude"],
            "address": bld["complete_address"],
            "existing_archetype": bld.get("archetype", ""),
            "existing_stories": bld.get("number_stories", ""),
            "existing_roof": bld.get("roof_shape_u", ""),
            "existing_walls": bld.get("construction_type_u", ""),
            "existing_status": bld.get("status_u", ""),
            "existing_steer_damage": bld.get("steer_damage_rating", ""),
        }

        # Predict archetype components from Street View
        if key in sv_embeds:
            emb = sv_embeds[key].reshape(1, -1)

            for target_name in ["stories", "roof", "walls"]:
                if target_name in classifiers:
                    clf, le = classifiers[target_name]
                    pred = le.inverse_transform(clf.predict(emb))[0]
                    prob = clf.predict_proba(emb).max()
                    result[f"predicted_{target_name}"] = pred
                    result[f"{target_name}_confidence"] = round(float(prob), 3)

            # Derive archetype
            ps = result.get("predicted_stories", "")
            pr = result.get("predicted_roof", "")
            pw = result.get("predicted_walls", "")
            if ps and pr:
                try:
                    stories_int = int(ps)
                except ValueError:
                    stories_int = 1
                walls_simplified = "brick" if "masonry" in str(pw) else str(pw)
                roof_simplified = pr.split("_")[0] if "_" in str(pr) else str(pr)
                arch, score = derive_archetype(stories_int, walls_simplified, roof_simplified)
                result["predicted_archetype"] = arch
                result["archetype_match_score"] = score

        # Predict damage from Nearmap
        if key in nm_embeds and "damage" in classifiers:
            emb = nm_embeds[key].reshape(1, -1)
            clf, le = classifiers["damage"]
            pred = le.inverse_transform(clf.predict(emb))[0]
            prob = clf.predict_proba(emb).max()
            result["predicted_damage_nearmap"] = pred
            result["damage_nm_confidence"] = round(float(prob), 3)

        # Predict damage from Street View (where pano is post-event)
        if key in sv_embeds and "damage_sv" in classifiers:
            emb = sv_embeds[key].reshape(1, -1)
            clf, le = classifiers["damage_sv"]
            pred = le.inverse_transform(clf.predict(emb))[0]
            prob = clf.predict_proba(emb).max()
            result["predicted_damage_sv"] = pred
            result["damage_sv_confidence"] = round(float(prob), 3)

        results[key] = result

    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    # --- Phase 4: Write output ---
    print("\n--- Phase 4: Writing output ---")
    out_keys = [
        "idx", "source", "lat", "lon", "address",
        "existing_archetype", "predicted_archetype", "archetype_match_score",
        "existing_stories", "predicted_stories", "stories_confidence",
        "existing_roof", "predicted_roof", "roof_confidence",
        "existing_walls", "predicted_walls", "walls_confidence",
        "existing_status", "existing_steer_damage",
        "predicted_damage_nearmap", "damage_nm_confidence",
        "predicted_damage_sv", "damage_sv_confidence",
    ]

    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_keys, extrasaction="ignore")
        w.writeheader()
        for key in sorted(results.keys(), key=lambda k: int(k)):
            w.writerow(results[key])

    n_arch = sum(1 for r in results.values() if r.get("predicted_archetype"))
    n_dmg = sum(1 for r in results.values()
                if r.get("predicted_damage_nearmap") or r.get("predicted_damage_sv"))
    print(f"\nDone. {n_arch} archetype predictions, {n_dmg} damage predictions.")
    print(f"Output: {OUT_PATH}")


if __name__ == "__main__":
    main()
