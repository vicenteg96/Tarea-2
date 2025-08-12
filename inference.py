from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# NOTA: el modelo fue entrenado con input (256,256,3) RGB y normalización /255.0
# y la primera salida es la clasificación de 2 clases (fresh=0, infected=1).
# Fuente: tu notebook (summary del modelo y función de preprocesamiento).  # ver README
INPUT_SIZE: Tuple[int, int] = (256, 256)
CLASS_MAP = {0: "fresh", 1: "infected"}

_model: Any = None

def load_model(path: str = None):
    global _model
    if _model is None:
        model_path = path or os.getenv("MODEL_PATH", "model/model_1.h5")
        _model = tf.keras.models.load_model(model_path)
    return _model


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.resize(INPUT_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:  # por si llegara en grises
        arr = np.stack([arr, arr, arr], axis=-1)
    return np.expand_dims(arr, axis=0)  # (1,256,256,3)

def predict_from_image(img: Image.Image) -> Dict[str, Any]:
    model = load_model()
    x = preprocess(img)
    outputs = model.predict(x, verbose=0)

    # model_1 tiene 3 salidas: [class_logits, bbox, mask]
    # Tomamos la primera para clasificación:
    class_probs = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    class_probs = class_probs[0]  # (2,)
    cls_idx = int(np.argmax(class_probs))
    return {
        "label": CLASS_MAP.get(cls_idx, str(cls_idx)),
        "score": float(class_probs[cls_idx]),
        "raw_probs": {CLASS_MAP[i]: float(class_probs[i]) for i in range(len(class_probs))}
    }
