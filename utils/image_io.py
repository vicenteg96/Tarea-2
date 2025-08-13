import io, base64, requests
from PIL import Image

def load_image_from_url(url: str) -> Image.Image:
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    return img

def load_image_from_base64(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return img

def image_to_base64(img: Image.Image, fmt: str = "JPEG", quality: int = 85) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def make_thumbnail_base64(img: Image.Image, max_size=(128, 128),
                          fmt: str = "JPEG", quality: int = 70) -> str:
    # copia para no mutar la imagen original
    thumb = img.copy()
    thumb.thumbnail(max_size)  # mantiene aspect ratio
    return image_to_base64(thumb, fmt=fmt, quality=quality)

def make_thumbnail_bytes(img, max_size=(128,128), fmt="JPEG", quality=70):
    import io
    from PIL import Image
    thumb = img.copy(); thumb.thumbnail(max_size)
    buf = io.BytesIO(); thumb.save(buf, format=fmt, quality=quality, optimize=True)
    return buf.getvalue()

