from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

# Cargar el modelo entrenado
modelo = joblib.load('random_forest_model.pkl')

# Crear la aplicación FastAPI
app = FastAPI()

# Clase para definir los datos de entrada
class InputData(BaseModel):
    datos: list

@app.post("/predict")
async def predict(input_data: InputData):
    # Convertir los datos en un array de NumPy
    datos_array = np.array(input_data.datos)
    
    # Hacer predicciones
    predicciones = modelo.predict(datos_array).tolist()
    
    return {"predicciones": predicciones}
