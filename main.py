# main.py
import time
import uuid
import binascii
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import UnidentifiedImageError

from schemas import ImageInput, PredictResponse
from utils.image_io import (
    load_image_from_url,
    load_image_from_base64,
    make_thumbnail_base64,
)
from inference import (
    load_model,
    predict_from_image,
    get_model_version,
    INPUT_SIZE,
    CLASSES,
)


app = FastAPI(
    title="Clasificador de Salmón",
    description="API que clasifica imágenes de salmón como fresh/infected",
    version="1.0.0",
)

# CORS abierto para facilitar pruebas externas
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    """Carga el modelo una sola vez al iniciar el servidor."""
    try:
        load_model()
        print("[startup] Modelo cargado OK")
    except Exception as e:
        # No rompemos el arranque; solo log
        print(f"[startup] No se pudo cargar el modelo: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def root():
    return {"ok": True, "message": "API de Clasificación de Salmón lista. Ver /docs para la especificación."}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: ImageInput):
    """
    Recibe image_url o image_base64 (exactamente uno), ejecuta predicción (argmax) y
    devuelve JSON con metadatos + miniatura en base64. Sin threshold.
    Respuestas de error claras cuando el archivo no es imagen o la URL/base64 es inválida.
    """
    t0 = time.perf_counter()

    # Validación de entrada: exactamente uno
    if bool(payload.image_url) == bool(payload.image_base64):
        raise HTTPException(
            status_code=422,
            detail="Debes enviar exactamente uno: 'image_url' o 'image_base64'.",
        )

    try:
        # Cargar imagen desde URL o base64
        if payload.image_url:
            img = load_image_from_url(str(payload.image_url))
            input_type = "image_url"
        else:
            try:
                img = load_image_from_base64(payload.image_base64)  # type: ignore
            except binascii.Error:
                # Base64 mal formado / padding incorrecto
                raise HTTPException(
                    status_code=400,
                    detail="El campo 'image_base64' no es válido (base64 mal formado).",
                )
            input_type = "image_base64"

    except UnidentifiedImageError:
        # PIL no pudo identificar el archivo como imagen
        raise HTTPException(
            status_code=400,
            detail="El archivo no es una imagen válida. Se espera JPG o PNG.",
        )
    except ValueError as e:
        # Cualquier ValueError claro desde los loaders
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Error genérico al cargar la imagen (incluye problemas de red en URL)
        raise HTTPException(
            status_code=400,
            detail=f"No se pudo procesar la imagen: {e}",
        )

    try:
        # Predicción (argmax) — asumimos que load_model() ya se llamó en startup
        raw = predict_from_image(img)  # {"cls_idx": int, "probs": [p0, p1]}
        probs = {CLASSES[i]: float(round(raw["probs"][i], 6)) for i in range(len(CLASSES))}
        label = CLASSES[raw["cls_idx"]]
        decision = label  # sin threshold: decision = label

        # Miniatura (base64) siempre
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
        raise HTTPException(
            status_code=400,
            detail=f"Error al ejecutar la predicción: {e}",
        )
