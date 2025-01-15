from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import pickle
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

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

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "random_forest_model.pkl")
data_path = os.path.join(base_dir, "datos_vivienda_clean.csv")

# Cargar modelo
if os.path.exists(model_path):
    with open(model_path, "rb") as file:
        model = pickle.load(file)
else:
    raise FileNotFoundError(f"El archivo del modelo '{model_path}' no se encuentra.")

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
    try:
        print("Datos recibidos para predicción:", request.dict())

        if request.Año <= 0:
            raise HTTPException(status_code=400, detail="El año debe ser positivo.")

        input_data = pd.DataFrame([request.dict()])
        input_data["Provincias_Cod"] = input_data["Provincias"].map(province_to_code)

        if input_data["Provincias_Cod"].isnull().any():
            raise HTTPException(status_code=400, detail="Provincia no válida.")

        input_data["Total_scaled"] = 0.5

        expected_columns = ["Provincias_Cod", "Año", "Mes", "Total_scaled"]
        missing_columns = [col for col in expected_columns if col not in input_data.columns]
        if missing_columns:
            raise HTTPException(status_code=400, detail=f"Faltan columnas: {', '.join(missing_columns)}")

        print("Datos procesados para el modelo:", input_data)

        prediction = model.predict(input_data)[0]
        return {"prediction": round(prediction, 2)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

