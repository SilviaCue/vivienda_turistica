from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import unicodedata
import logging
import joblib
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

logging.basicConfig(level=logging.INFO)
logging.info("Servidor iniciado correctamente.")

# Función de normalización
def normalize(text):
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8').lower()

# Mapeo de provincias a códigos
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
model_path = os.path.join(base_dir, "gradient_boosting_model.pkl")
data_path = os.path.join(base_dir, "datos_vivienda_clean.csv")

# Cargar el modelo
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        logging.info("Modelo cargado exitosamente.")
        logging.info(f"Tipo del modelo cargado: {type(model)}")
        logging.info(f"Columnas esperadas por el modelo: {model.feature_names_in_}")
    except Exception as e:
        logging.error(f"Error al cargar el modelo: {e}")
        raise FileNotFoundError("No se pudo cargar el modelo.")
else:
    raise FileNotFoundError(f"El archivo del modelo '{model_path}' no se encuentra.")

# Probar el modelo al iniciar
try:
    test_data = pd.DataFrame([{
        "Provincias_Cod": 28,  # 
        "Año": 2080,
        "Mes": 5,
        "Total_scaled": 0.8
    }])

    # Verificar las columnas esperadas por el modelo
    expected_columns = list(model.feature_names_in_)
    if not all(col in test_data.columns for col in expected_columns):
        raise ValueError(f"Las columnas de entrada no coinciden. Se esperaban: {expected_columns}, pero se encontraron: {test_data.columns.tolist()}")

    # Reorganizar las columnas según lo esperado por el modelo
    test_data = test_data[expected_columns]

    # Realizar predicción de prueba
    test_prediction = model.predict(test_data)[0]
    logging.info(f"Prueba del modelo exitosa. Predicción: {test_prediction}")
except Exception as e:
    logging.error(f"Error durante la prueba del modelo: {e}")
    raise

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

 # Registrar los datos enviados al modelo
    logging.info(f"Datos enviados al modelo: {input_data}")


@app.post("/predict")
def predict(request: PredictionRequest):
    logging.info("El endpoint /predict fue llamado.")
    logging.info(f"Datos recibidos del usuario: {request.dict()}")
    try:
        # Crear un DataFrame con los datos recibidos
        input_data = pd.DataFrame([request.dict()])

        # Normalizar y mapear provincia
        input_data["Provincias_Cod"] = input_data["Provincias"].apply(normalize).map(province_to_code_normalized)

        # Validar si la provincia es válida
        if input_data["Provincias_Cod"].isnull().any():
            logging.error("Provincia no válida.")
            raise HTTPException(status_code=400, detail="Provincia no válida. Verifique los datos ingresados.")

        # Calcular `Total_scaled` basado en datos históricos
        mean_scaled = data.loc[
            (data["Provincias_Cod"] == input_data["Provincias_Cod"].iloc[0]) &
            (data["Mes"] == input_data["Mes"].iloc[0]),
            "Total_scaled"
        ].mean()

        # Usar la media calculada o un valor predeterminado si no hay datos
        input_data["Total_scaled"] = mean_scaled if not pd.isnull(mean_scaled) else 0.5

        # Verificar las columnas esperadas por el modelo
        expected_columns = list(model.feature_names_in_)
        if not all(col in input_data.columns for col in expected_columns):
            raise HTTPException(status_code=400, detail=f"Las columnas de entrada no coinciden. Se esperaban: {expected_columns}")

        # Reorganizar las columnas según lo esperado por el modelo
        input_data = input_data[expected_columns]

        # Registrar los datos enviados al modelo
        logging.info(f"Datos enviados al modelo: {input_data}")

        # Realizar la predicción
        prediction = model.predict(input_data)[0]
        logging.info(f"Predicción realizada: {prediction}")
        return {"prediction": round(prediction, 2)}
    except Exception as e:
        logging.error(f"Error durante la predicción: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
   
