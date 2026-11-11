"""
train_pipeline.py — builds the colour/texture pipeline models the backend votes with.

Run once, on the same machine and Python that runs the backend, so the saved
models always match the installed library versions:

    py -3.11 train_pipeline.py --features path\\to\\features

`--features` is a folder holding the cached feature files written by the
Colab notebooks into  Food Sure/fair_comparison/ :

    feats_black_pepper_powder_trainval_gw0.npz   feats_black_pepper_powder_test_gw0.npz
    feats_red_chilli_powder_trainval_gw1.npz     feats_red_chilli_powder_test_gw1.npz
    feats_turmeric_trainval_gw0.npz              feats_turmeric_test_gw0.npz

Each model is trained on the train+val photos only (the same photos ResNet50
learned from) and scored on the held-out test photos, so the test set stays
untouched. Settings are unchanged from the team notebook "Untitled5.ipynb":
3 classes (pure / adulterant A / adulterant B), RobustScaler ->
SelectKBest(f_classif, 65) -> CatBoost, or the CatBoost + LightGBM +
ExtraTrees soft vote for red chilli.

Writes pipeline_models/<spice>/bundle.joblib and metadata.json.
"""

import os
import re
import json
import argparse
import warnings

import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPICES = ["black_pepper_powder", "red_chilli_powder", "turmeric"]
GRAY_WORLD_SPICES = {"red_chilli_powder"}


def to_type(label):
    """'10_pct_rice_flour' -> 'rice_flour', 'pure' -> 'pure'"""
    label = str(label).lower()
    return "pure" if label == "pure" else re.sub(r"^\d+_pct_", "", label)


def make_clf(spice):
    import catboost as cb
    if spice in GRAY_WORLD_SPICES:  # Pipeline 2
        import lightgbm as lgb
        from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
        m1 = cb.CatBoostClassifier(iterations=550, learning_rate=0.035, depth=5, auto_class_weights="Balanced",
                                   verbose=0, random_seed=42, allow_writing_files=False)
        m2 = lgb.LGBMClassifier(n_estimators=350, learning_rate=0.035, max_depth=5, class_weight="balanced",
                                random_state=42, verbose=-1)
        m3 = ExtraTreesClassifier(n_estimators=300, max_depth=12, class_weight="balanced", random_state=42)
        return VotingClassifier(estimators=[("cb", m1), ("lgb", m2), ("et", m3)], voting="soft")
    return cb.CatBoostClassifier(iterations=600, learning_rate=0.04, depth=5, auto_class_weights="Balanced",
                                 verbose=0, random_seed=42, allow_writing_files=False)  # Pipeline 1


def load_part(features_dir, spice, part):
    gw = int(spice in GRAY_WORLD_SPICES)
    path = os.path.join(features_dir, f"feats_{spice}_{part}_gw{gw}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing feature file: {path}")
    d = np.load(path, allow_pickle=True)
    y3 = np.array([to_type(l) for l in d["y"]])
    return d["X"].astype(np.float32), y3, d["groups"].astype(int)


def train_spice(spice, features_dir, out_dir):
    from sklearn.preprocessing import RobustScaler
    from sklearn.feature_selection import SelectKBest, f_classif

    X_tr, y_tr, g_tr = load_part(features_dir, spice, "trainval")
    X_te, y_te, g_te = load_part(features_dir, spice, "test")

    scaler = RobustScaler().fit(X_tr)
    selector = SelectKBest(score_func=f_classif, k=min(65, X_tr.shape[1])).fit(scaler.transform(X_tr), y_tr)
    clf = make_clf(spice)
    clf.fit(selector.transform(scaler.transform(X_tr)), y_tr)
    classes = [str(c) for c in clf.classes_]

    # Photo-level score on the held-out test photos (crop probabilities averaged per photo)
    probs = np.asarray(clf.predict_proba(selector.transform(scaler.transform(X_te))))
    ids = np.unique(g_te)
    photo_probs = np.stack([probs[g_te == i].mean(axis=0) for i in ids])
    photo_true = np.array([y_te[g_te == i][0] for i in ids])
    photo_pred = np.array(classes)[photo_probs.argmax(axis=1)]
    acc = float((photo_pred == photo_true).mean() * 100)

    spice_dir = os.path.join(out_dir, spice)
    os.makedirs(spice_dir, exist_ok=True)
    joblib.dump({"scaler": scaler, "selector": selector, "clf": clf, "classes": classes},
                os.path.join(spice_dir, "bundle.joblib"), compress=3)
    np.savez(os.path.join(spice_dir, "test_photo_probs.npz"), ids=ids, probs=photo_probs,
             true=photo_true, classes=np.array(classes))
    meta = {
        "spice": spice,
        "classes": classes,
        "architecture": "Pipeline 2: CatBoost + LightGBM + ExtraTrees soft vote (gray-world)"
        if spice in GRAY_WORLD_SPICES else "Pipeline 1: CatBoost",
        "trained_on": "train+val photos",
        "n_train_photos": int(len(np.unique(g_tr))),
        "n_test_photos": int(len(ids)),
        "test_acc_3class": round(acc, 2),
    }
    with open(os.path.join(spice_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{spice:<22} trained on {meta['n_train_photos']} photos | "
          f"3-class test accuracy {acc:.2f}% on {meta['n_test_photos']} photos")
    return meta


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True, help="folder with the feats_*.npz files")
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "pipeline_models"))
    args = ap.parse_args()
    for s in SPICES:
        train_spice(s, args.features, args.out)
    print("Saved to", args.out)
