# schemas.py
from pydantic import BaseModel, HttpUrl, field_validator
from typing import Optional, Dict, List, Literal

class ImageInput(BaseModel):
    image_url: Optional[HttpUrl] = None
    image_base64: Optional[str] = None
    # Mantener por compatibilidad, pero se IGNORA en el servidor
    threshold: Optional[float] = None

    @field_validator("image_base64")
    @classmethod
    def strip_base64(cls, v):
        if v is None:
            return v
        return v.split(",")[-1].strip()

    def model_post_init(self, __context):
        if bool(self.image_url) == bool(self.image_base64):
            raise ValueError("Debes enviar exactamente uno: 'image_url' o 'image_base64'.")

class PredictResponse(BaseModel):
    ok: bool
    request_id: str
    model_version: str
    took_ms: int

    # salida principal
    label: str
    score: float
    decision: str  # = label (sin threshold)
    probs: Dict[str, float]
    classes: List[str]
    image_size: List[int]
    input_type: Literal["image_url", "image_base64"]

    # eco/soporte
    image_url: Optional[HttpUrl] = None
    image_base64: Optional[str] = None
    image_thumb_base64: Optional[str] = None

    # evitar warning por 'model_version'
    model_config = {
        "protected_namespaces": ()
    }
