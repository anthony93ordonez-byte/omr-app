from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import json

app = FastAPI(title="Motor OMR Multicolumna Calibrado")

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
    """Detecta los 4 cuadros negros y normaliza la imagen a 1000x1414 px (proporción A4)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centros = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        area = cv2.contourArea(c)
        if len(approx) == 4 and area > 400:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.7 <= aspect_ratio <= 1.3:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    centros.append([cX, cY])

    if len(centros) < 4:
        raise ValueError("No se detectaron las 4 esquinas negras. Asegúrate de encuadrar la hoja completa.")

    pts_arr = np.array(centros)
    rect = ordenar_puntos(pts_arr)

    ancho_dst, alto_dst = 1000, 1414
    dst = np.array([
        [95, 95],
        [905, 95],
        [905, 1319],
        [95, 1319]
    ], dtype="float32")

    matriz = cv2.getPerspectiveTransform(rect, dst)
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
        total_preguntas = len(lista_clave)

        contents = await imagen.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        enderezada = alinear_hoja(img)
        
        thresh = cv2.adaptiveThreshold(
            enderezada, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 31, 10
        )

        # Conversión mm a píxeles: factor_x = 4.762 | factor_y = 4.761
        start_y = int(58.0 * 4.761)  # 276 px
        row_h = 8.2 * 4.761          # 39.0 px

        if total_preguntas <= 25:
            start_x = int(38.0 * 4.762)     # 181 px
            ancho_columna = 0
            col_w = 9.8 * 4.762             # 46.7 px
            radio = 13.5
        elif total_preguntas <= 50:
            start_x = int(36.0 * 4.762)     # 171 px
            ancho_columna = int(78.0 * 4.762)  # 371 px
            col_w = 8.5 * 4.762             # 40.5 px
            radio = 12.5
        elif total_preguntas <= 75:
            start_x = int(32.0 * 4.762)     # 152 px
            ancho_columna = int(56.0 * 4.762)  # 266 px
            col_w = 6.8 * 4.762             # 32.4 px
            radio = 10.5
        else:
            # 4 Columnas (hasta 100 preguntas) calibradas con la nueva holgura
            start_x = int(27.0 * 4.762)        # 129 px
            ancho_columna = int(43.5 * 4.762)  # 207 px
            col_w = 6.4 * 4.762                # 30.5 px
            radio = 10.0

        buenas, malas, blancas = 0, 0, 0
        respuestas_detectadas = []

        for i, resp_correcta in enumerate(lista_clave):
            col_idx = i // 25
            fila_idx = i % 25

            y = int(start_y + (fila_idx * row_h))
            base_x_pregunta = int(start_x + (col_idx * ancho_columna))

            marcas = []
            for j, opc in enumerate(lista_opciones):
                x = int(base_x_pregunta + (j * col_w))
                
                mascara = np.zeros(thresh.shape, dtype="uint8")
                cv2.circle(mascara, (x, y), int(radio), 255, -1)
                
                total_pixeles = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mascara))
                marcas.append((total_pixeles, opc))

            marcas.sort(key=lambda x: x[0], reverse=True)
            max_pix, seleccion = marcas[0]
            segundo_pix = marcas[1][0] if len(marcas) > 1 else 0

            factor_area = (radio / 13.5) ** 2
            umbral_blanco = 58 * factor_area
            umbral_doble = 30 * factor_area

            if max_pix < umbral_blanco:
                blancas += 1
                respuestas_detectadas.append("BLANCO")
            elif (max_pix - segundo_pix) < umbral_doble:
                malas += 1
                respuestas_detectadas.append("DOBLE")
            elif seleccion == resp_correcta:
                buenas += 1
                respuestas_detectadas.append(seleccion)
            else:
                malas += 1
                respuestas_detectadas.append(seleccion)

        nota = round((buenas / total_preguntas) * 20, 2)

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