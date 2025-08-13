# client.py
import base64, json, time, os, requests

BASE = os.getenv("API_URL", "https://tarea-2-4ivb.onrender.com")

def pretty(d): 
    print(json.dumps(d, indent=2, ensure_ascii=False))

def req_predict(payload, title):
    url = f"{BASE}/predict"
    t0 = time.perf_counter()
    r = requests.post(url, json=payload, timeout=60)
    dt = int((time.perf_counter() - t0)*1000)
    print(f"\n== {title} ==")
    print("POST", url)
    print("Payload:")
    pretty(payload)
    print(f"Status: {r.status_code}  |  Took: {dt} ms")
    try:
        pretty(r.json())
    except Exception:
        print(r.text[:500])

def load_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def main():
    # 1) Predicción por URL (válida)
    req_predict(
        {
            "image_url": "https://www.gastronomiavasca.net/uploads/image/file/3268/salmon.jpg",
            "threshold": 0.5
        },
        "Req #1: image_url válida"
    )

    # 2) Predicción por Base64 (válida)
    # Pon aquí una imagen local pequeña (por ejemplo ./samples/salmon.jpg)
    img_path = os.getenv("LOCAL_IMG", "samples/salmon.jpg")
    if os.path.exists(img_path):
        b64 = load_b64(img_path)
        req_predict(
            {
                "image_base64": b64,
                "threshold": 0.6
            },
            "Req #2: image_base64 local (umbral 0.6)"
        )
    else:
        print(f"\n[AVISO] No se encontró {img_path}. Omite Req #2 o fija LOCAL_IMG.")

    # 3) Caso de error: enviar ambos campos a la vez (debe fallar con mensaje claro)
    both_payload = {
        "image_url": "https://www.gastronomiavasca.net/uploads/image/file/3268/salmon.jpg",
        "image_base64": "AAAA",  # dummy
    }
    req_predict(both_payload, "Req #3: error (url y base64 a la vez)")

if __name__ == "__main__":
    main()

