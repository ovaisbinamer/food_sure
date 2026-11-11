"""
test_predict.py — catches silent accuracy regressions (preprocessing, resize,
voting) by pinning today's answers on the six held-out sample photos.

    py -3.11 test_predict.py

The expected values are what the models answer now, wrong answers included
(two of the six are wrong). Retrained models change them on purpose; then
re-run this, check the new answers, and update EXPECTED.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from fastapi.testclient import TestClient

import main
import model_utils

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "foodsure-web", "public", "samples")

# (photo, predicted_class, voted confidence)
EXPECTED = [
    ("black_pepper_powder__10_pct_charcoal", "10_pct_charcoal", 0.9673),
    ("black_pepper_powder__15_pct_millet", "15_pct_millet", 0.636),
    ("red_chilli_powder__10_pct_sawdust", "15_pct_semolina", 0.5183),
    ("red_chilli_powder__15_pct_semolina", "15_pct_semolina", 0.5906),
    ("turmeric__10_pct_rice_flour", "10_pct_rice_flour", 0.9317),
    ("turmeric__5_pct_semolina", "5_pct_rice_flour", 0.4688),
]


def test_parse_severity():
    assert model_utils.parse_severity("pure") == ("none", "PURE", 0)
    assert model_utils.parse_severity("10_pct_rice_flour") == ("rice_flour", "MEDIUM", 10)
    assert model_utils.parse_severity("junk") == ("unknown", "UNKNOWN", None)


def test_sample_photos_unchanged():
    for name, expected_class, expected_conf in EXPECTED:
        with open(os.path.join(SAMPLES_DIR, name + ".jpg"), "rb") as f:
            r = model_utils.predict_adulteration(f.read(), name.split("__")[0])
        assert r["decision"] == "vote", f"{name}: colour/texture model did not vote ({r['decision']})"
        assert r["predicted_class"] == expected_class, f"{name}: {r['predicted_class']} != {expected_class}"
        assert abs(r["confidence"] - expected_conf) < 0.01, f"{name}: confidence {r['confidence']} != {expected_conf}"


def test_api_rejects_bad_uploads():
    client = TestClient(main.app)
    not_a_photo = client.post("/predict", data={"spice": "turmeric"},
                              files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert not_a_photo.status_code == 400, not_a_photo.text
    wrong_type = client.post("/predict", data={"spice": "turmeric"},
                             files={"file": ("x.txt", b"hello", "text/plain")})
    assert wrong_type.status_code == 400, wrong_type.text


if __name__ == "__main__":
    for test in (test_parse_severity, test_sample_photos_unchanged, test_api_rejects_bad_uploads):
        test()
        print("ok", test.__name__)
