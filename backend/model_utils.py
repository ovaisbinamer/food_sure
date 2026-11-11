"""
model_utils.py — Food-Sure inference layer.

Loads the models exported by notebook Cell 13 (web_models/<spice>/model.keras
+ label_map.json, plus web_models/manifest.json) and exposes
predict_adulteration(), the one function the FastAPI app (and any future
Flask/Streamlit frontend) calls.

Two models answer together: ResNet50 (this file) and the colour/texture
pipeline (pipeline_utils.py). They vote on which adulterant is present;
ResNet50 then gives the level. See predict_adulteration() for the details.

Confirmed against the real notebook (Cell 13's web-export + Cell 14's
predict_adulteration()) as of the 2026-08-16 training run, where ResNet50
won all three spices:
  1. web_models/<spice>/ always contains exactly 'model.keras' and
     'label_map.json' (fixed filenames — not stored in the manifest).
  2. Class labels are '{pct}_pct_{adulterant}', e.g. '10_pct_sawdust',
     '5_pct_millet', plus the literal 'pure'.
  3. manifest.json's "preprocess" field is one of 'rescale' (Custom CNN),
     'mobilenet_v2', or 'resnet50' — matching tf.keras.applications'
     preprocess_input for the winning model, or raw /255 rescale for the CNN.
"""

import os
import json
import re
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Point this at wherever you copied web_models/ locally after downloading
# it from Google Drive. Override with the WEB_MODELS_DIR env var if needed.
WEB_MODELS_DIR = os.environ.get(
    "WEB_MODELS_DIR", os.path.join(BASE_DIR, "web_models")
)

SEVERITY_MAP = {
    "pure": "PURE",
    "5": "LOW",
    "10": "MEDIUM",
    "15": "HIGH",
}

# Each spice model is a closed 7-class classifier (pure + 2 adulterants x 3
# concentrations) trained ONLY on that spice. Nothing in the pipeline checks
# that the uploaded photo actually matches the spice the user selected in the
# UI -- if it doesn't, the model still forces the image into whichever of its
# 7 known classes looks closest, and returns a confident-looking but
# meaningless result (e.g. a red chilli photo run against the turmeric model
# coming back as "5% semolina", a turmeric-only adulterant). Low confidence
# does NOT catch this: on the sample photos only 1 of 12 wrong-spice runs fell
# under the threshold (a turmeric photo run as red chilli came back "pure",
# 92%), and neither mean colour nor distance from the training data separated
# right from wrong spice. So the threshold only flags unclear photos, and the
# UI tells the user the spice choice must be right.
# shortcut: no wrong-spice detection; needs a spice classifier trained on the original photos (Drive).
# Override with the LOW_CONFIDENCE_THRESHOLD env var if needed.
LOW_CONFIDENCE_THRESHOLD = float(os.environ.get("LOW_CONFIDENCE_THRESHOLD", "0.45"))

# The vote on "which adulterant": share of the final opinion given to ResNet50;
# the rest goes to the colour/texture pipeline. 0.5 = plain average, fixed in
# advance (not tuned on the test photos).
VOTE_WEIGHT_RESNET = float(os.environ.get("VOTE_WEIGHT_RESNET", "0.5"))

# The models were trained on samples mixed at exactly 5, 10 and 15%. A sample
# in between is reported as the nearest of those, so each level is shown as
# the band of values closest to it rather than as an exact number.
CONCENTRATION_BANDS = {
    "LOW": "under 7.5%",
    "MEDIUM": "7.5-12.5%",
    "HIGH": "above 12.5%",
}

_manifest = None
_models = {}       # spice -> tf.keras.Model
_label_maps = {}   # spice -> {index(str): class_name}


def _load_manifest():
    global _manifest
    if _manifest is None:
        manifest_path = os.path.join(WEB_MODELS_DIR, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"manifest.json not found at {manifest_path}. "
                f"Run Cell 13 in the notebook and copy web_models/ here first."
            )
        with open(manifest_path) as f:
            _manifest = json.load(f)
    return _manifest


def get_available_spices():
    return list(_load_manifest().keys())


def _get_model(spice):
    manifest = _load_manifest()
    if spice not in manifest:
        raise ValueError(f"Unknown spice '{spice}'. Available: {list(manifest.keys())}")

    if spice not in _models:
        entry = manifest[spice]
        # Cell 13 always writes these two fixed filenames per spice folder —
        # the manifest itself doesn't carry a model_file/label_map_file key.
        model_path = os.path.join(WEB_MODELS_DIR, spice, "model.keras")
        label_path = os.path.join(WEB_MODELS_DIR, spice, "label_map.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file missing: {model_path}")

        _models[spice] = tf.keras.models.load_model(model_path)
        with open(label_path) as f:
            _label_maps[spice] = json.load(f)

    return _models[spice], _label_maps[spice], manifest[spice]


def parse_severity(class_name: str):
    """
    Turns a class label like '10_pct_charcoal' into (adulterant, severity, pct).
    'pure' -> ('none', 'PURE', 0).

    Real label format confirmed from the notebook (Cell 8/9 generator classes
    and Cell 14's class_to_severity()): '{pct}_pct_{adulterant}', e.g.
    '5_pct_millet', '10_pct_sawdust', '15_pct_rice_flour'. Severity is decided
    by substring/pct match, mirroring the notebook's own logic exactly.
    """
    name = class_name.strip().lower()
    if name == "pure":
        return "none", "PURE", 0

    match = re.match(r"^(\d+)_pct_(.+)$", name)
    if match:
        pct, adulterant = match.group(1), match.group(2)
        severity = SEVERITY_MAP.get(pct, "UNKNOWN")
        return adulterant, severity, int(pct)

    return "unknown", "UNKNOWN", None


def _preprocess_image(pil_img: Image.Image, img_size: int, preprocess: str):
    """
    IMPORTANT: for 'mobilenet_v2' and 'resnet50', do NOT call keras.applications'
    preprocess_input here. Cell 8/9's build_mobilenetv2()/build_resnet50() bake
    preprocessing into the model graph itself as the very first layer
    (`x = mobilenet_preprocess(inputs)` / `x = resnet_preprocess(inputs)`, right
    after the Input layer, before the frozen base model). The notebook's own
    Cell 7 comment confirms this deliberately: "plain_datagen_tl = ImageDataGenerator()
    # no preprocessing — model handles it" and warns that applying preprocess_input
    a second time outside the model "would double-process the pixel values and
    hurt accuracy." Verified empirically: applying preprocess_input here dropped
    validation accuracy from ~75% (matching the real test_acc) to ~19% on a
    21-image spot-check across all three spices — this was a real, serious bug,
    not just theoretical. So for those two tags we only resize/float-cast and
    hand the model raw [0,255] pixels; the model does the rest internally.
    Only 'rescale' (Custom CNN, which has no internal preprocessing layer)
    needs the external /255 scaling done here.
    """
    # Phone cameras commonly store an EXIF orientation tag instead of
    # physically rotating pixels. PIL ignores that tag by default, so a
    # portrait photo can get fed in sideways/upside-down at inference time
    # even though every training image (loaded via the dataset pipeline,
    # which strips this) was upright. This alone can tank real-world demo
    # accuracy on phone-shot samples even though the model is fine.
    pil_img = ImageOps.exif_transpose(pil_img)
    # Resize with NEAREST, not PIL's default (bicubic). Training and every
    # reported test accuracy used Keras load_img(target_size=...), whose
    # default interpolation is 'nearest'. Checked on 12 held-out test photos:
    # with NEAREST the probabilities match the Colab evaluation exactly; with
    # the default they drift by up to 0.69 on a single photo and can change
    # the answer.
    pil_img = pil_img.convert("RGB").resize((img_size, img_size), Image.NEAREST)
    arr = np.array(pil_img).astype("float32")
    x = np.expand_dims(arr, axis=0)

    if preprocess == "rescale":
        x = x / 255.0
    elif preprocess in ("mobilenet_v2", "resnet50"):
        pass  # model's own input layer applies the matching preprocess_input
    else:
        raise ValueError(f"Unknown preprocess type: {preprocess}")

    return x


def type_of(class_name: str) -> str:
    """'10_pct_charcoal' -> 'charcoal', 'pure' -> 'pure' (the 3-class view)."""
    adulterant, _, _ = parse_severity(class_name)
    return "pure" if adulterant == "none" else adulterant


def predict_adulteration(image, spice: str) -> dict:
    """
    image: file path (str), bytes, or PIL.Image
    spice: one of get_available_spices()

    How the answer is decided:
      1. WHICH ADULTERANT (pure / adulterant A / adulterant B) -- a vote.
         ResNet50's 7 class probabilities are added up per adulterant, the
         colour/texture pipeline (pipeline_utils.py) gives its own 3
         probabilities, and the two are averaged (VOTE_WEIGHT_RESNET, 50/50
         by default). If the pipeline model for this spice is not installed
         the answer is ResNet50's alone and "decision" says so.
      2. HOW MUCH -- ResNet50's most likely level (5/10/15) for the adulterant
         chosen in step 1, reported as a band (see CONCENTRATION_BANDS).

    Returns the same keys as before (spice, predicted_class, confidence,
    adulterant, severity, concentration_pct, severity_message, model_used,
    top3, low_confidence, warning) plus:
      concentration_range  e.g. "7.5-12.5%" (None when pure)
      level_confidence     ResNet50's confidence in the level, given the adulterant
      decision             "vote" or "resnet_only"
      models_agree         True/False, or None when only ResNet50 answered
      type_votes           [{type, voted, resnet50, pipeline}] sorted by voted
    "confidence" is the voted probability of the chosen adulterant.
    """
    model, label_map, entry = _get_model(spice)

    if isinstance(image, (str, bytes, os.PathLike)) and not isinstance(image, bytes):
        pil_img = Image.open(image)
    elif isinstance(image, bytes):
        import io
        pil_img = Image.open(io.BytesIO(image))
    elif isinstance(image, Image.Image):
        pil_img = image
    else:
        raise TypeError("image must be a file path, bytes, or PIL.Image")
    pil_img.load()

    x = _preprocess_image(pil_img, entry["img_size"], entry["preprocess"])
    preds = model.predict(x, verbose=0)[0]  # shape (num_classes,)
    labels = [label_map[str(i)] for i in range(len(preds))]
    class_types = [type_of(l) for l in labels]

    # --- Step 1: which adulterant -------------------------------------
    resnet_types = {}
    for t, p in zip(class_types, preds):
        resnet_types[t] = resnet_types.get(t, 0.0) + float(p)

    pipeline_types = None
    try:
        import pipeline_utils
        pipeline_types = pipeline_utils.predict_types(pil_img, spice)
    except Exception as e:  # a missing library or a bad bundle must not take the API down
        print(f"[pipeline] skipped for {spice}: {type(e).__name__}: {e}")
    if pipeline_types is not None and set(pipeline_types) != set(resnet_types):
        print(f"[pipeline] class mismatch for {spice}: {sorted(pipeline_types)} vs {sorted(resnet_types)}")
        pipeline_types = None

    if pipeline_types is not None:
        w = VOTE_WEIGHT_RESNET
        voted = {t: w * resnet_types[t] + (1 - w) * pipeline_types[t] for t in resnet_types}
        decision = "vote"
        models_agree = max(resnet_types, key=resnet_types.get) == max(pipeline_types, key=pipeline_types.get)
        model_used = f"{entry['model_name']} + colour/texture pipeline"
    else:
        voted = dict(resnet_types)
        decision = "resnet_only"
        models_agree = None
        model_used = entry["model_name"]

    chosen_type = max(voted, key=voted.get)
    confidence = float(voted[chosen_type])

    # --- Step 2: how much (ResNet50's level for the chosen adulterant) ---
    idx = [i for i, t in enumerate(class_types) if t == chosen_type]
    top_idx = max(idx, key=lambda i: preds[i])
    predicted_class = labels[top_idx]
    type_mass = float(sum(preds[i] for i in idx))
    level_confidence = float(preds[top_idx]) / type_mass if type_mass > 0 else 0.0

    adulterant, severity, pct = parse_severity(predicted_class)
    concentration_range = CONCENTRATION_BANDS.get(severity)
    nice = adulterant.replace("_", " ")

    severity_messages = {
        "PURE": f"Sample appears pure {spice.replace('_', ' ')}.",
        "LOW": f"Low-level adulteration detected ({nice}, {concentration_range}).",
        "MEDIUM": f"Moderate adulteration detected ({nice}, {concentration_range}).",
        "HIGH": f"High adulteration detected ({nice}, {concentration_range}).",
        "UNKNOWN": "Could not determine severity from model output.",
    }

    order = np.argsort(preds)[::-1]
    top3 = [[labels[int(i)], float(preds[int(i)])] for i in order[:3]]  # ResNet50's own 7-class view

    type_votes = [
        {
            "type": t,
            "voted": round(float(voted[t]), 4),
            "resnet50": round(float(resnet_types[t]), 4),
            "pipeline": round(float(pipeline_types[t]), 4) if pipeline_types is not None else None,
        }
        for t in sorted(voted, key=voted.get, reverse=True)
    ]

    low_confidence = confidence < LOW_CONFIDENCE_THRESHOLD
    warning = (
        f"This result has unusually low confidence ({confidence * 100:.0f}%). "
        "The photo may be unclear. Try a clearer, well-lit shot."
        if low_confidence
        else None
    )

    return {
        "spice": spice,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "adulterant": adulterant,
        "severity": severity,
        "concentration_pct": pct,
        "concentration_range": concentration_range,
        "level_confidence": round(level_confidence, 4),
        "severity_message": severity_messages.get(severity, severity_messages["UNKNOWN"]),
        "model_used": model_used,
        "decision": decision,
        "models_agree": models_agree,
        "type_votes": type_votes,
        "top3": top3,
        "low_confidence": low_confidence,
        "warning": warning,
    }
