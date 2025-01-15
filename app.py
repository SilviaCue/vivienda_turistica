from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import pickle
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Cargar modelo
model_path = "random_forest_model.pkl"
if os.path.exists(model_path):
    with open(model_path, "rb") as file:
        model = pickle.load(file)
else:
    raise FileNotFoundError(f"El archivo del modelo '{model_path}' no se encuentra.")

# Cargar datos preprocesados
data_path = "datos_vivienda_clean.csv"
if os.path.exists(data_path):
    data = pd.read_csv(data_path)
else:
    raise FileNotFoundError(f"El archivo de datos '{data_path}' no se encuentra.")

class PredictionRequest(BaseModel):
    Comunidad: str
    Provincias_Cod: int
    Año: int
    Mes: int

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        input_data = pd.DataFrame([request.dict()])
        prediction = model.predict(input_data)
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))