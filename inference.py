from typing import Dict, Any, Tuple
import os, hashlib
import numpy as np
from PIL import Image
import tensorflow as tf

# Entrada esperada por el modelo y clases
INPUT_SIZE: Tuple[int, int] = (256, 256)
CLASSES = ["fresh", "infected"]
CLASS_MAP = {i: c for i, c in enumerate(CLASSES)}

_model = None
_model_version: str | None = None  # se setea al cargar el modelo


def _compute_model_version(path: str) -> str:
    """
    Devuelve un identificador estable de versión para el artefacto (.h5),
    usando nombre de archivo + hash corto del contenido.
    """
    try:
        with open(path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()[:12]
        return f"{os.path.basename(path)}#{h}"
    except Exception:
        return os.path.basename(path)


def load_model(path: str = None):
    """
    Carga el modelo .h5 en modo tolerante a formatos legacy:
    - compile=False: evita reconstituir pérdidas/métricas antiguas.
    - safe_mode=False: permite deserializar aunque no coincida 1:1.
    - custom_objects: capa Conv2DTranspose explícita por compatibilidad.
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
        _model_version = _compute_model_version(model_path)
    return _model


def get_model_version() -> str:
    """Expuesto para incluir versión en la respuesta de la API."""
    return _model_version or "model_1.h5"


def preprocess(img: Image.Image) -> np.ndarray:
    """
    - Redimensiona a 256x256
    - Convierte a float32 y normaliza /255
    - Garantiza 3 canales (RGB)
    - Devuelve batch (1, H, W, 3)
    """
    img = img.resize(INPUT_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:  # imagen en escala de grises -> duplicar a 3 canales
        arr = np.stack([arr, arr, arr], axis=-1)
    return np.expand_dims(arr, axis=0)


def predict_from_image(img: Image.Image) -> Dict[str, Any]:
    """
    Ejecuta la predicción y devuelve:
      - cls_idx: índice de clase con mayor probabilidad
      - probs: lista [p_fresh, p_infected]
    Notas:
      model_1.h5 produce 3 salidas [class_probs, bbox, mask]; usamos la primera.
    """
    model = load_model()
    x = preprocess(img)
    outputs = model.predict(x, verbose=0)

    class_probs = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    class_probs = class_probs[0]  # (2,)
    cls_idx = int(np.argmax(class_probs))

    return {
        "cls_idx": cls_idx,
        "probs": class_probs.tolist(),
    }
