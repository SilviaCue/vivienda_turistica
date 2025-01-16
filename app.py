from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import pickle
import os
import unicodedata  
import logging


app = FastAPI()
templates = Jinja2Templates(directory="templates")

logging.basicConfig(level=logging.INFO)
logging.info("Servidor iniciado correctamente.")

# Función de normalización
def normalize(text):
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8').lower()

# Mapeo de provincias a códigos (completo)
province_to_code = {
    "01 Araba/Álava": 1,
    "02 Albacete": 2,
    "03 Alicante/Alacant": 3,
    "04 Almería": 4,
    "05 Ávila": 5,
    "06 Badajoz": 6,
    "07 Balears, Illes": 7,
    "08 Barcelona": 8,
    "09 Burgos": 9,
    "10 Cáceres": 10,
    "11 Cádiz": 11,
    "12 Castellón/Castelló": 12,
    "13 Ciudad Real": 13,
    "14 Córdoba": 14,
    "15 Coruña, A": 15,
    "16 Cuenca": 16,
    "17 Girona": 17,
    "18 Granada": 18,
    "19 Guadalajara": 19,
    "20 Gipuzkoa": 20,
    "21 Huelva": 21,
    "22 Huesca": 22,
    "23 Jaén": 23,
    "24 León": 24,
    "25 Lleida": 25,
    "26 Rioja, La": 26,
    "27 Lugo": 27,
    "28 Madrid": 28,
    "29 Málaga": 29,
    "30 Murcia": 30,
    "31 Navarra": 31,
    "32 Ourense": 32,
    "33 Asturias": 33,
    "34 Palencia": 34,
    "35 Palmas, Las": 35,
    "36 Pontevedra": 36,
    "37 Salamanca": 37,
    "38 Santa Cruz de Tenerife": 38,
    "39 Cantabria": 39,
    "40 Segovia": 40,
    "41 Sevilla": 41,
    "42 Soria": 42,
    "43 Tarragona": 43,
    "44 Teruel": 44,
    "45 Toledo": 45,
    "46 Valencia/València": 46,
    "47 Valladolid": 47,
    "48 Bizkaia": 48,
    "49 Zamora": 49,
    "50 Zaragoza": 50,
    "51 Ceuta": 51,
    "52 Melilla": 52
}

# Normalizar las claves del mapeo
province_to_code_normalized = {normalize(key): value for key, value in province_to_code.items()}

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "random_forest_model.pkl")
data_path = os.path.join(base_dir, "datos_vivienda_clean.csv")

# Cargar modelo
if os.path.exists(model_path):
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    # Prueba del modelo al iniciar
    test_data = pd.DataFrame([{
        "Provincias_Cod": 28,  # Código para Madrid
        "Año": 2023,
        "Mes": 5,
        "Total_scaled": 0.5
    }])
    try:
        test_prediction = model.predict(test_data)[0]
        logging.info(f"Prueba del modelo exitosa. Predicción: {test_prediction}")
    except Exception as e:
        logging.error(f"Error al probar el modelo: {e}")
else:
    raise FileNotFoundError(f"El archivo del modelo '{model_path}' no se encuentra.")



# Cargar datos preprocesados
if os.path.exists(data_path):
    data = pd.read_csv(data_path)
else:
    raise FileNotFoundError(f"El archivo de datos '{data_path}' no se encuentra.")

# Modelo de entrada para predicción
class PredictionRequest(BaseModel):
    Provincias: str
    Año: int
    Mes: int

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict(request: PredictionRequest):
    logging.info("El endpoint /predict fue llamado.")
    try:
        # Crear un nuevo DataFrame con los datos recibidos
        input_data = pd.DataFrame([request.dict()])
        logging.info(f"Datos recibidos: {input_data}")

        # Normalizar y mapear provincia a código
        input_data["Provincias_Cod"] = input_data["Provincias"].apply(normalize).map(province_to_code_normalized)

        # Validar si alguna provincia es inválida
        if input_data["Provincias_Cod"].isnull().any():
            logging.error("Provincia no válida.")
            raise HTTPException(status_code=400, detail="Provincia no válida.")

        # Agregar columna requerida por el modelo
        input_data["Total_scaled"] = 0.5

        # Validar columnas
        expected_columns = ["Provincias_Cod", "Año", "Mes", "Total_scaled"]
        logging.info(f"Datos procesados para el modelo: {input_data[expected_columns]}")

        # Realizar predicción
        prediction = model.predict(input_data[expected_columns])[0]
        logging.info(f"Predicción realizada: {prediction}")

        return {"prediction": round(prediction, 2)}
    except Exception as e:
        logging.error(f"Error durante la predicción: {e}")
        raise HTTPException(status_code=400, detail=str(e))

