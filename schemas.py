from pydantic import BaseModel, HttpUrl, field_validator
from typing import Optional

class ImageInput(BaseModel):
    image_url: Optional[HttpUrl] = None
    image_base64: Optional[str] = None

    @field_validator("image_base64")
    @classmethod
    def strip_base64(cls, v):
        if v is None:
            return v
        return v.split(",")[-1].strip()  # permite 'data:image/...;base64,XXXX'

    # Validación cruzada: exactamente uno de los dos campos
    def model_post_init(self, __context):
        if bool(self.image_url) == bool(self.image_base64):
            raise ValueError("Debes enviar exactamente uno: 'image_url' o 'image_base64'.")
