from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from schemas import ImageInput
from utils.image_io import load_image_from_url, load_image_from_base64
from inference import load_model, predict_from_image

app = FastAPI(
    title="Salmon Health Classifier (model_1.h5)",
    description="API para predecir si un salmón está sano (fresh) o infectado (infected) desde una foto.",
    version="1.0.0",
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

@app.post("/predict")
def predict(payload: ImageInput):
    try:
        if payload.image_url:
            img = load_image_from_url(str(payload.image_url))
        else:
            img = load_image_from_base64(payload.image_base64)  # type: ignore

        result = predict_from_image(img)
        return JSONResponse(content={"ok": True, "result": result})

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
