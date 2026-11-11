# Food-Sure Backend

Serves your trained models to a web frontend. This is separate from the
Colab notebook, which only trains and evaluates models — it never runs
as a live server.

## How the pieces fit together

1. **Colab notebook** (`FOODSURE_COMPLETE_BACKEND.ipynb`) trains 5 models
   per spice and saves checkpoints + results JSON to Drive.
2. **`cell13_export_models.py`** (paste into the notebook as a new cell)
   picks the best deep-learning model per spice and exports it to
   `web_models/<spice>/model.keras` + `label_map.json`, plus a top-level
   `manifest.json`.
3. **This backend** (`model_utils.py` + `main.py`) loads `web_models/`
   and serves predictions over HTTP. Your frontend (whatever you build in
   October — Flask/Streamlit/React) calls this API; it never touches
   TensorFlow directly.

## Setup (local)

1. Run Cell 13 in Colab. This creates `web_models/` inside your
   `Food Sure` Drive folder.
2. Download `web_models/` from Drive to your machine, and place it next
   to `main.py` so the layout looks like:
   ```
   backend/
     main.py
     model_utils.py
     requirements.txt
     web_models/
       manifest.json
       black_pepper_powder/
         model.keras
         label_map.json
       red_chilli_powder/
         model.keras
         label_map.json
       turmeric/
         model.keras
         label_map.json
   ```
   (Or set `WEB_MODELS_DIR` env var to point elsewhere instead of moving files.)
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the server:
   ```
   uvicorn main:app --reload --port 8000
   ```
5. Open `http://127.0.0.1:8000/docs` — Swagger UI lets you upload a test
   image and call `/predict` directly, no frontend needed yet.

## Testing with curl

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "spice=black_pepper_powder" \
  -F "file=@/path/to/test_image.jpg"
```

## Known things to check before this actually works

- **`parse_severity()` in `model_utils.py`** assumes class folder names
  look like `pure`, `charcoal_5`, `charcoal_10`, `millet_15`, etc. If your
  real dataset folder names use a different pattern, that's the one
  function to fix.
- **`_preprocess_image()`** assumes MobileNetV2 uses
  `mobilenet_v2.preprocess_input`. If ResNet50 ends up being the winning
  model for a spice, swap in `resnet50.preprocess_input` for that case —
  currently it's hardcoded to mobilenet's preprocessing for all
  `keras_app`-type models.
- Cell 13 currently only exports CustomCNN / MobileNetV2 / ResNet50
  (deep learning models), not SVM / Random Forest — those need the
  101-dim GLCM+color feature vector instead of a raw image, so they
  don't fit a simple "upload a photo" web app without also shipping the
  feature extractor and scaler. Say the word if you want SVM/RF wired in
  too.

## Deploying later (Render/Railway-style)

- Add a `Procfile` or start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Commit `web_models/` (or fetch it from cloud storage on startup — Git
  isn't great for `.keras` files if they're large).
- Tighten `allow_origins` in `main.py`'s CORS config to your actual
  frontend domain instead of `"*"`.


## Two-model vote (added November 2026)

`/predict` now asks two models and combines them:

1. **Which adulterant** (pure / adulterant A / adulterant B): ResNet50's 7
   class probabilities are added up per adulterant and averaged 50/50 with
   the colour/texture pipeline's 3 probabilities (`pipeline_utils.py`).
2. **How much**: ResNet50's most likely level for the chosen adulterant,
   shown as a band: low = under 7.5%, medium = 7.5-12.5%, high = above 12.5%.

Extra response fields: `concentration_range`, `level_confidence`, `decision`
(`"vote"` or `"resnet_only"`), `models_agree`, `type_votes`. `confidence` is
now the voted probability of the chosen adulterant.

The pipeline models are not in Drive; build them once on this machine:

```
py -3.11 train_pipeline.py --features path	oeatures
```

where the folder holds the six `feats_*.npz` files from
`Food Sure/fair_comparison/` in Drive. This writes
`pipeline_models/<spice>/bundle.joblib`. Until that exists (or if a pipeline
library is missing) the API answers with ResNet50 alone and says so in
`decision`.

`numpy` must stay on 1.26.x for TensorFlow 2.17. Installing the pipeline
libraries without the pins in `requirements.txt` upgrades numpy and breaks
TensorFlow.
