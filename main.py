# main.py
import time
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import ImageInput, PredictResponse
from utils.image_io import load_image_from_url, load_image_from_base64, make_thumbnail_base64
from inference import load_model, predict_from_image, get_model_version, INPUT_SIZE, CLASSES

# Crear la app FastAPI
app = FastAPI(
    title="Clasificador de Pescado",
    description="API para clasificar imágenes de pescado como fresh/infected",
    version="1.0.0"
)

# Middleware CORS para permitir peticiones desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    """Carga el modelo al iniciar el servidor."""
    load_model()
    print("[startup] Modelo cargado OK")

@app.get("/")
def root():
    return {"ok": True, "message": "API de Clasificación de Pescado lista"}

@app.post("/predict", response_model=PredictResponse)
def predict(payload: ImageInput):
    """
    Recibe image_url o image_base64, ejecuta predicción (argmax)
    y devuelve JSON con metadatos + miniatura en base64. Sin threshold.
    """
    try:
        t0 = time.perf_counter()
        input_type = "image_url" if payload.image_url else "image_base64"

        # 1) Cargar imagen
        if payload.image_url:
            img = load_image_from_url(str(payload.image_url))
        else:
            img = load_image_from_base64(payload.image_base64)  # type: ignore

        # 2) Predicción (argmax)
        load_model()  # asegura modelo en memoria
        raw = predict_from_image(img)  # {"cls_idx": int, "probs": [p0, p1]}
        probs = {CLASSES[i]: float(round(raw["probs"][i], 6)) for i in range(len(CLASSES))}
        label = CLASSES[raw["cls_idx"]]

        # 3) decision = label (sin threshold)
        decision = label

        # 4) Miniatura (base64)
        image_thumb_base64 = make_thumbnail_base64(img)

        took_ms = int((time.perf_counter() - t0) * 1000)
        return PredictResponse(
            ok=True,
            request_id=str(uuid.uuid4()),
            model_version=get_model_version(),
            took_ms=took_ms,

            label=label,
            score=probs[label],
            decision=decision,
            probs=probs,
            classes=CLASSES,
            image_size=[INPUT_SIZE[0], INPUT_SIZE[1]],
            input_type=input_type,

            image_url=str(payload.image_url) if payload.image_url else None,
            image_base64=(payload.image_base64 if payload.image_base64 else None),
            image_thumb_base64=image_thumb_base64,
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
