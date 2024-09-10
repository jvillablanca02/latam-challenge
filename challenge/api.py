from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Cargar el modelo previamente entrenado
model = joblib.load("ruta_a_tu_modelo.pkl")

# Definir el esquema de entrada
class FlightData(BaseModel):
    OPERA: str
    MES: int
    TIPOVUELO: str

# Endpoint de salud
@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}

# Endpoint de predicción
@app.post("/predict", status_code=200)
async def post_predict(data: FlightData) -> dict:
    try:
        # Convertir los datos de entrada a un DataFrame
        input_data = pd.DataFrame([data.dict()])
        
        # Preprocesar los datos de entrada
        input_data = pd.get_dummies(input_data)
        
        # Asegurarse de que las columnas coincidan con las del modelo
        model_columns = joblib.load("ruta_a_tus_columnas.pkl")
        input_data = input_data.reindex(columns=model_columns, fill_value=0)
        
        # Realizar la predicción
        prediction = model.predict(input_data)
        
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
