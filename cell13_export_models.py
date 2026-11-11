# ============================================================
# CELL 13 — Export winning model per spice to web_models/
# Paste this as a NEW cell in FOODSURE_COMPLETE_BACKEND.ipynb,
# after Cell 22/24 (once all 5 models have been trained and
# results/*_results.json + checkpoints_v2/ exist).
#
# ASSUMPTION (adjust if your real filenames differ):
#   checkpoints_v2/{spice}_{ModelName}_best.keras
#   results/{spice}_{ModelName}_results.json  containing at least:
#     {"test_accuracy": 0.71, "class_names": ["pure", "charcoal_5", ...]}
#   ModelName in {"SVM","RandomForest","CustomCNN","MobileNetV2","ResNet50"}
#   (SVM/RandomForest are .pkl not .keras — handled separately below,
#   but only CNN-family models are exported for the web app since the
#   web app takes raw images, not precomputed 101-dim feature vectors.)
# ============================================================

import os, json, glob, shutil

DRIVE_ROOT   = "/content/drive/MyDrive/Food Sure"
CKPT_DIR     = os.path.join(DRIVE_ROOT, "checkpoints_v2")
RESULTS_DIR  = os.path.join(DRIVE_ROOT, "results")
WEB_MODELS   = os.path.join(DRIVE_ROOT, "web_models")

# Only these are usable by an image-upload web app (they take raw images).
# SVM / RandomForest need the 101-dim GLCM+color feature vector, which is
# a separate pipeline — skip them here unless you also want to ship the
# feature extractor + scaler + .pkl model to the backend (more moving parts).
DEEP_MODEL_CANDIDATES = ["CustomCNN", "MobileNetV2", "ResNet50"]

os.makedirs(WEB_MODELS, exist_ok=True)

manifest = {}

for spice in SPICES:  # SPICES comes from Cell 4 config, e.g. ["black_pepper_powder","red_chilli_powder","turmeric"]
    best_model_name = None
    best_acc = -1.0
    best_class_names = None

    for model_name in DEEP_MODEL_CANDIDATES:
        results_path = os.path.join(RESULTS_DIR, f"{spice}_{model_name}_results.json")
        if not os.path.exists(results_path):
            print(f"[skip] no results for {spice} / {model_name}")
            continue
        with open(results_path) as f:
            r = json.load(f)
        acc = r.get("test_accuracy", -1)
        if acc > best_acc:
            best_acc = acc
            best_model_name = model_name
            best_class_names = r.get("class_names")

    if best_model_name is None:
        print(f"[WARNING] No results found for spice={spice}. Skipping export.")
        continue

    if best_class_names is None:
        raise ValueError(
            f"'{spice}_{best_model_name}_results.json' has no 'class_names' key. "
            f"Fix the Custom CNN cell (12) to save class_names into its results JSON, "
            f"then re-run this cell."
        )

    # Locate the checkpoint file (fallback scan if naming differs slightly)
    ckpt_path = os.path.join(CKPT_DIR, f"{spice}_{best_model_name}_best.keras")
    if not os.path.exists(ckpt_path):
        candidates = glob.glob(os.path.join(CKPT_DIR, f"*{spice}*{best_model_name}*.keras"))
        if not candidates:
            raise FileNotFoundError(
                f"Could not find a checkpoint for {spice}/{best_model_name} in {CKPT_DIR}. "
                f"Expected '{spice}_{best_model_name}_best.keras' or similar."
            )
        ckpt_path = candidates[0]

    spice_dir = os.path.join(WEB_MODELS, spice)
    os.makedirs(spice_dir, exist_ok=True)

    dest_model_path = os.path.join(spice_dir, "model.keras")
    shutil.copy2(ckpt_path, dest_model_path)

    label_map = {i: name for i, name in enumerate(best_class_names)}
    with open(os.path.join(spice_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    # img size depends on which model won
    img_size = 128 if best_model_name == "CustomCNN" else 224
    preprocess = "custom_cnn" if best_model_name == "CustomCNN" else "keras_app"

    manifest[spice] = {
        "model_file": "model.keras",
        "label_map_file": "label_map.json",
        "model_name": best_model_name,
        "test_accuracy": best_acc,
        "img_size": img_size,
        "preprocess": preprocess,  # tells the backend which preprocessing fn to use
    }

    print(f"[OK] {spice}: exported {best_model_name} (test_acc={best_acc:.4f}) -> {dest_model_path}")

manifest_path = os.path.join(WEB_MODELS, "manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nWrote manifest: {manifest_path}")
print(json.dumps(manifest, indent=2))
