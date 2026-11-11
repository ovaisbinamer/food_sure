"""
main.py — Food-Sure FastAPI backend.

Run locally:
    uvicorn main:app --reload --port 8000

Then open http://127.0.0.1:8000/docs for interactive Swagger UI
(useful for testing and for showing  the API live).

Endpoints:
  GET  /health              -> liveness check
  GET  /spices               -> list of spices this backend can classify
  POST /predict               -> upload an image + spice, get adulteration result
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import UnidentifiedImageError

import model_utils

app = FastAPI(
    title="Food-Sure API",
    description="AI-based food adulteration detection for turmeric, red chilli, and black pepper.",
    version="1.0.0",
)

# Allow your frontend (Flask/Streamlit/React, running on a different port)
# to call this API from the browser. Tighten allow_origins before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/spices")
def list_spices():
    try:
        spices = model_utils.get_available_spices()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"spices": spices}


# Plain def, not async: the models block for seconds, and FastAPI runs a def
# route in a worker thread so /health keeps answering meanwhile.
@app.post("/predict")
def predict(
    spice: str = Form(..., description="One of: black_pepper_powder, red_chilli_powder, turmeric"),
    file: UploadFile = File(..., description="Photo of the spice sample"),
):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = file.file.read()

    try:
        result = model_utils.predict_adulteration(image_bytes, spice)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not read that file as a photo. Try a JPG or PNG.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return result
