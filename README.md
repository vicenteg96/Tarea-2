# Clasificador de Imágenes de Salmon (Fresh vs Infected)

Este proyecto implementa una API en FastAPI que clasifica imágenes de salmon como fresh (fresco) o infected (infectado) utilizando un modelo preentrenado en TensorFlow/Keras.

La API recibe una imagen (vía URL o archivo local) y devuelve:

- La clase predicha (fresh o infected)
- La probabilidad asociada a cada clase
- Una miniatura de la imagen codificada en base64

Esta API está desarrollada con FastAPI y actualmente se encuentra desplegada en Render.

Se puede acceder a la aplicación en producción en la siguiente URL:

https://tarea-2-4ivb.onrender.com/

Docs:

https://tarea-2-4ivb.onrender.com/docs

Health status:

https://tarea-2-4ivb.onrender.com/health -> {"status": "ok"}

## Estructura del proyecto

Tarea-2/
│

├── app/                # Código principal FastAPI

│   └── main.py

├── models/             # Carga del modelo 

│   └── model_1.h5      

│

│── utils/              # Funciones auxiliares (imágenes, base64, etc.)

│   └── image_io.py 

│
├── Test/               # Carpeta de pruebas locales

│   └── test_data/      # Imágenes de prueba

│   └── client.py       # Cliente Python con la función predict_any

│   └──Test.ipynb       #Prueba en Render

│   └──Test_local.ipynb #Prueba en local

│

├── requirements.txt    # Dependencias

├── inference.py        # Lógica de predicción del modelo.

├── render.yaml         # Configuración para desplegar en Render.

├── schemas.py          # Definición de entrada y salida (contrato de la API).

└── README.md           # Documentación



## Estructura del JSON de entrada

El endpoint /predict acepta exactamente uno de los siguientes campos:

image_url: enlace público a la imagen (formato JPG o PNG).

image_base64: cadena base64 de la imagen (opcional, usado internamente).

Nota: El usuario no necesita generar un base64. Para simplificar, este repo incluye la función predict_any que permite usar directamente rutas locales o URLs.

## Uso desde Python

Ejemplo de uso con la función predict_any incluida en client.py:

Dependencias

Las dependencias del proyecto se encuentran en requirements.txt.

Render las instala automáticamente durante el despliegue.

Codigo Python:

from client import predict_any

### Caso 1: Imagen desde URL

resp = predict_any(image_url="https://www.gastronomiavasca.net/uploads/image/file/3268/salmon.jpg")

print(resp)

### Caso 2: Imagen local (ruta absoluta)

resp = predict_any(image_path=r"C:/Users/galla/OneDrive/Documentos/GitHub/Tarea-2/Test/test_data/test.png")

print(resp)

Cuando usas la función predict_any, obtendrás en consola algo así:

HTTP 200

Label: fresh

Decision: fresh

Score: 0.995426

Probabilidades: {'fresh': 0.995426, 'infected': 0.004574}

(Imagen en miniatura)

Para ver un ejemplo claro, favor revisar el archivo test.ipynb ubicado en /Test

El despliegue está activo en Render, no es necesario correr nada localmente.

Las pruebas locales solo se usaron para verificar el modelo antes de publicarlo, pero no forman parte del flujo oficial de uso.

Para interactuar, se recomienda usar el cliente Python (client.py) o directamente probar la API en /docs.

