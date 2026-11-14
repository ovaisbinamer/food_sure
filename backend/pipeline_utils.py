"""
pipeline_utils.py — the colour/texture pipeline (second opinion for the vote).

This is the classifier stage of the team's handcrafted-feature pipeline
(Colab notebook "Untitled5.ipynb"): each photo is cut into four 256px crops,
152 colour and texture numbers are computed per crop, and a classifier says
pure / adulterant A / adulterant B. Crop probabilities are averaged per photo.

  * black pepper, turmeric : RobustScaler -> SelectKBest(65) -> CatBoost
  * red chilli             : gray-world white balance, then a soft vote of
                             CatBoost + LightGBM + ExtraTrees

The feature extractor and model settings are copied unchanged from that
notebook so numbers measured there carry over. Only the classifier is used
here; the notebook's percentage regressor is not (on the held-out test photos
its answer was no closer to the truth than always answering 10%).

Models live in pipeline_models/<spice>/bundle.joblib and are produced by
train_pipeline.py. If a bundle is missing, model_utils falls back to ResNet50
alone, so the API keeps working.
"""

import os
import io
import json

import numpy as np
from PIL import Image, ImageOps

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_MODELS_DIR = os.environ.get(
    "PIPELINE_MODELS_DIR", os.path.join(BASE_DIR, "pipeline_models")
)

CROP_SIZE = 256
CROPS_PER_IMAGE = 4
N_FEATURES = 152
GRAY_WORLD_SPICES = {"red_chilli_powder"}

_bundles = {}  # spice -> dict(scaler, selector, clf, classes) or None if unavailable


# --------------------------------------------------------------------------
# Feature extraction (unchanged from the notebook)
# --------------------------------------------------------------------------
def extract_features(img_bgr, use_gray_world=False):
    import cv2
    from scipy.stats import skew, kurtosis
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

    if use_gray_world:
        b, g, r = cv2.split(img_bgr.astype(np.float32))
        gray_avg = (np.mean(b) + np.mean(g) + np.mean(r)) / 3.0
        b = np.clip(b * (gray_avg / (np.mean(b) + 1e-5)), 0, 255)
        g = np.clip(g * (gray_avg / (np.mean(g) + 1e-5)), 0, 255)
        r = np.clip(r * (gray_avg / (np.mean(r) + 1e-5)), 0, 255)
        img_bgr = cv2.merge([b, g, r]).astype(np.uint8)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    feats = []

    # A. Colour moments across 12 channels (mean, std, skew, kurtosis)
    for space in [img_rgb, img_hsv, img_lab, img_ycrcb]:
        for ch in range(3):
            plane = space[:, :, ch].astype(np.float32).ravel()
            mean_v, std_v = np.mean(plane), np.std(plane)
            sk_v = float(skew(plane)) if std_v > 1e-3 else 0.0
            kt_v = float(kurtosis(plane)) if std_v > 1e-3 else 0.0
            feats.extend([mean_v, std_v, sk_v, kt_v])

    # B. Normalised histograms (H, S, L*, a*, b*)
    h_hist, _ = np.histogram(img_hsv[:, :, 0], bins=16, range=(0, 180), density=True)
    s_hist, _ = np.histogram(img_hsv[:, :, 1], bins=16, range=(0, 256), density=True)
    l_hist, _ = np.histogram(img_lab[:, :, 0], bins=16, range=(0, 256), density=True)
    a_hist, _ = np.histogram(img_lab[:, :, 1], bins=16, range=(0, 256), density=True)
    b_hist, _ = np.histogram(img_lab[:, :, 2], bins=16, range=(0, 256), density=True)
    feats.extend(np.hstack([h_hist, s_hist, l_hist, a_hist, b_hist]))

    # C. Domain physical indices
    lab_l = img_lab[:, :, 0].astype(np.float32)
    lab_a = img_lab[:, :, 1].astype(np.float32)
    lab_b = img_lab[:, :, 2].astype(np.float32)
    hsv_s = img_hsv[:, :, 1].astype(np.float32)
    cai = np.mean(lab_b) / (np.mean(lab_l) + 1e-5)
    rpi = np.mean(lab_a) / (np.mean(hsv_s) + 1e-5)
    fleck_ratio = np.sum(img_gray > 140) / (img_gray.size + 1e-5)
    lap_var = float(np.var(cv2.Laplacian(img_gray, cv2.CV_64F)))
    feats.extend([cai, rpi, fleck_ratio, lap_var])

    # D. Isotropic GLCM
    glcm = graycomatrix(img_gray, distances=[1, 2], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=256, symmetric=True, normed=True)
    glcm_iso = np.mean(glcm, axis=3, keepdims=True)
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        feats.extend(graycoprops(glcm_iso, prop).ravel())

    # E. LBP texture
    lbp = local_binary_pattern(img_gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
    feats.extend(lbp_hist)

    return np.array(feats, dtype=np.float32)


def image_crops(img_bgr):
    """Four 256px crops at native resolution inside the central spice region."""
    import cv2

    h, w = img_bgr.shape[:2]
    if h >= CROP_SIZE and w >= CROP_SIZE:
        cy, cx = h // 2, w // 2
        # Stay within the central 50% region so outer paper/plate borders are excluded
        half_span_y = min(h // 4, max(0, cy - CROP_SIZE // 2))
        half_span_x = min(w // 4, max(0, cx - CROP_SIZE // 2))
        coords = [
            (cy - CROP_SIZE // 2, cx - CROP_SIZE // 2),
            (max(0, cy - CROP_SIZE // 2 - half_span_y), max(0, cx - CROP_SIZE // 2 - half_span_x)),
            (min(h - CROP_SIZE, cy - CROP_SIZE // 2 + half_span_y), min(w - CROP_SIZE, cx - CROP_SIZE // 2 + half_span_x)),
            (max(0, cy - CROP_SIZE // 2 - half_span_y), min(w - CROP_SIZE, cx - CROP_SIZE // 2 + half_span_x)),
        ]
        return [img_bgr[t:t + CROP_SIZE, l:l + CROP_SIZE] for t, l in coords[:CROPS_PER_IMAGE]]
    return [cv2.resize(img_bgr, (CROP_SIZE, CROP_SIZE))]


def pil_to_bgr(pil_img):
    """
    Same pixels the notebook's cv2.imread() would give: upright (EXIF rotation
    applied, as cv2.imread does by default), 3 channels, BGR order.
    """
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    return np.ascontiguousarray(np.array(pil_img)[:, :, ::-1])


def photo_features(pil_img, spice):
    use_gw = spice in GRAY_WORLD_SPICES
    return np.stack([extract_features(c, use_gray_world=use_gw) for c in image_crops(pil_to_bgr(pil_img))])


# --------------------------------------------------------------------------
# Model loading and prediction
# --------------------------------------------------------------------------
def _load_bundle(spice):
    if spice in _bundles:
        return _bundles[spice]
    path = os.path.join(PIPELINE_MODELS_DIR, spice, "bundle.joblib")
    bundle = None
    if os.path.exists(path):
        import joblib
        bundle = joblib.load(path)
        bundle["classes"] = [str(c) for c in bundle["classes"]]
    _bundles[spice] = bundle
    return bundle


def is_available(spice):
    return _load_bundle(spice) is not None


def get_metadata(spice):
    path = os.path.join(PIPELINE_MODELS_DIR, spice, "metadata.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def predict_types(pil_img, spice):
    """
    Returns {type_name: probability} for the photo, e.g.
    {'pure': 0.08, 'charcoal': 0.81, 'millet': 0.11}, or None if this spice
    has no pipeline model installed.
    """
    bundle = _load_bundle(spice)
    if bundle is None:
        return None
    X = photo_features(pil_img, spice)
    Xs = bundle["selector"].transform(bundle["scaler"].transform(X))
    probs = np.asarray(bundle["clf"].predict_proba(Xs)).mean(axis=0)
    return {c: float(p) for c, p in zip(bundle["classes"], probs)}


def predict_types_from_bytes(image_bytes, spice):
    return predict_types(Image.open(io.BytesIO(image_bytes)), spice)
