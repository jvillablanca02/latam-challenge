from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Cargar el modelo entrenado
model = joblib.load("challenge/finalized_model.pkl")

# Crear la aplicación FastAPI
app = FastAPI()

# Definir el modelo de datos para la solicitud de predicción
class PredictionRequest(BaseModel):
    features: list

# Endpoint para verificar el estado de la API
@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}

# Endpoint para realizar predicciones
@app.post("/predict", status_code=200)
async def post_predict(request: PredictionRequest) -> dict:
    features = np.array(request.features).reshape(1, -1)
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}

# Ejecutar la aplicación con Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

