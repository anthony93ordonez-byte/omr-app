from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import json

app = FastAPI(title="Motor OMR Calibrado Hoja Original")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def ordenar_puntos(pts):
    """Ordena los 4 marcadores: [sup-izq, sup-der, inf-der, inf-izq]"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def alinear_hoja(img):
    """Detecta los 4 cuadros negros de las esquinas y proyecta la hoja a 1000x1400 px"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centros = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        # Filtra cuadriláteros que correspondan a los cuadros negros de las esquinas
        if len(approx) == 4 and cv2.contourArea(c) > 250:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centros.append([cX, cY])

    if len(centros) < 4:
        raise ValueError("No se detectaron los 4 cuadros de las esquinas. Asegúrate de enfocar la hoja completa con buena luz.")

    centros = sorted(centros, key=lambda p: (p[0]**2 + p[1]**2))
    pts = ordenar_puntos(np.array(centros[:4], dtype="float32"))

    ancho_dst, alto_dst = 1000, 1400
    dst = np.array([
        [0, 0],
        [ancho_dst - 1, 0],
        [ancho_dst - 1, alto_dst - 1],
        [0, alto_dst - 1]
    ], dtype="float32")

    matriz = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(gray, matriz, (ancho_dst, alto_dst))

@app.post("/api/calificar")
async def calificar(
    imagen: UploadFile = File(...),
    clave: str = Form(...),
    opciones: str = Form(...)
):
    try:
        lista_clave = json.loads(clave)
        lista_opciones = json.loads(opciones)

        contents = await imagen.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 1. Enderezar la hoja con perspectiva
        enderezada = alinear_hoja(img)
        
        # 2. Umbralización adaptativa para evitar problemas de sombras
        thresh = cv2.adaptiveThreshold(
            enderezada, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 25, 9
        )

        # Coordenadas exactas para la hoja de la primera versión (startY = 45 mm)
        start_x = 166
        start_y = 212
        row_h = 42.4
        col_w = 47.1
        radio = 12

        buenas, malas, blancas = 0, 0, 0
        respuestas_detectadas = []

        for i, resp_correcta in enumerate(lista_clave):
            columna = 0 if i < 25 else 1
            fila = i if i < 25 else i - 25

            y = int(start_y + (fila * row_h))
            x_base = start_x if columna == 0 else start_x + 355

            marcas = []
            for j, opc in enumerate(lista_opciones):
                x = int(x_base + (j * col_w))
                
                mascara = np.zeros(thresh.shape, dtype="uint8")
                cv2.circle(mascara, (x, y), radio, 255, -1)
                
                total_pixeles = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mascara))
                marcas.append((total_pixeles, opc))

            marcas.sort(key=lambda x: x[0], reverse=True)
            max_pix, seleccion = marcas[0]
            segundo_pix = marcas[1][0] if len(marcas) > 1 else 0

            # Umbral de detección calibrado para marcas normales
            if max_pix < 50:
                blancas += 1
                respuestas_detectadas.append("BLANCO")
            elif (max_pix - segundo_pix) < 25:
                malas += 1
                respuestas_detectadas.append("DOBLE")
            elif seleccion == resp_correcta:
                buenas += 1
                respuestas_detectadas.append(seleccion)
            else:
                malas += 1
                respuestas_detectadas.append(seleccion)

        total = len(lista_clave)
        nota = round((buenas / total) * 20, 2)

        return {
            "status": "ok",
            "buenas": buenas,
            "malas": malas,
            "blancas": blancas,
            "puntaje": nota,
            "respuestas": respuestas_detectadas
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")