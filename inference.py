from typing import Dict, Any, Tuple
import os, hashlib
import numpy as np
from PIL import Image
import tensorflow as tf

INPUT_SIZE: Tuple[int, int] = (256, 256)
CLASSES = ["fresh", "infected"]
CLASS_MAP = {i: c for i, c in enumerate(CLASSES)}

_model = None
_model_version: str | None = None  

def _compute_model_version(path: str) -> str:
    # usa hash corto del archivo para versionar de forma estable
    try:
        with open(path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()[:12]
        return f"{os.path.basename(path)}#{h}"
    except Exception:
        return os.path.basename(path)

def load_model(path: str = None):
    """
    Carga tolerante para .h5 legacy
    """
    global _model, _model_version
    if _model is None:
        model_path = path or os.getenv("MODEL_PATH", "model/model_1.h5")
        _model = tf.keras.models.load_model(
            model_path,
            compile=False,
            safe_mode=False,
            custom_objects={"Conv2DTranspose": tf.keras.layers.Conv2DTranspose},
        )
        _model_version = _compute_model_version(model_path)  # <-- NUEVO
    return _model

def get_model_version() -> str: 
    return _model_version or "model_1.h5"

def preprocess(img: Image.Image) -> np.ndarray:
    img = img.resize(INPUT_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr

