import time, uuid
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from schemas import ImageInput, PredictResponse
from utils.image_io import load_image_from_url, load_image_from_base64, make_thumbnail_base64
from inference import load_model, predict_from_image, get_model_version, INPUT_SIZE, CLASSES


from schemas import ImageInput, PredictResponse
from utils.image_io import (
    load_image_from_url,
    load_image_from_base64,
    make_thumbnail_base64,
)
from inference import (
    load_model,
    predict_from_image,
    get_model_version,   # si no lo tienes, puedes devolver "model_1.h5"
    INPUT_SIZE,
    CLASSES,
)

app = FastAPI(
    title="Salmon Health Classifier (model_1.h5)",
    description="API para predecir si un salmón está sano (fresh) o infectado (infected) desde una foto.",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"],
)

@app.on_event("startup")
def _warmup():
    try:
        load_model()
        print("[startup] Modelo cargado OK")
    except Exception as e:
        print(f"[startup] No se pudo cargar el modelo: {e}")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.post("/predict", response_model=PredictResponse)
def predict(payload: ImageInput):
    """
    Recibe image_url o image_base64 (exactamente uno), ejecuta predicción y
    devuelve JSON con metadatos + miniatura en base64.
    """
    try:
        t0 = time.perf_counter()
        input_type = "image_url" if payload.image_url else "image_base64"

        # 1) Cargar imagen
        if payload.image_url:
            img = load_image_from_url(str(payload.image_url))
        else:
            img = load_image_from_base64(payload.image_base64)  # type: ignore

        # 2) Predicción
        load_model()  # asegura que el modelo esté cargado
        raw = predict_from_image(img)  # {"cls_idx": int, "probs": [p0,p1]}
        probs = {CLASSES[i]: float(round(raw["probs"][i], 6)) for i in range(len(CLASSES))}
        label = CLASSES[raw["cls_idx"]]

        # 3) Decisión con umbral
        threshold = float(payload.threshold if payload.threshold is not None else 0.5)
        decision = "infected" if probs["infected"] >= threshold else "fresh"

        # 4) Miniatura (base64) siempre
        image_thumb_base64 = make_thumbnail_base64(img)

        took_ms = int((time.perf_counter() - t0) * 1000)

        # 5) Respuesta tipada (deja a FastAPI serializar)
        return PredictResponse(
            ok=True,
            request_id=str(uuid.uuid4()),
            model_version=get_model_version(),
            took_ms=took_ms,

            label=label,
            score=probs[label],
            decision=decision,
            threshold=threshold,
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