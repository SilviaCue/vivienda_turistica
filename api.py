from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
import pandas as pd

# Cargar el modelo entrenado
modelo = joblib.load('random_forest_model.pkl')

# Crear la aplicación FastAPI
app = FastAPI()

# Clase para definir los datos de entrada
class InputData(BaseModel):
    datos: list

@app.post("/predict")
def predict(region: int, mes: int, año: int):
    # Crear entrada para el modelo
    entrada = pd.DataFrame([{
        "Provincias_Cod": region,
        "Total_scaled": 0.7,  # Ajustar si tienes valores reales
        "Año": año,
        "Mes": mes
    }])
    # Hacer predicciones
    prediccion = rf_model.predict(entrada)
    return {"prediccion": prediccion[0]}

