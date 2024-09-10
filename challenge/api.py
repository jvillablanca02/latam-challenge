import uvicorn
from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# Cargar los modelos entrenados
xgb_model = joblib.load("xgb_model.pkl")
logistic_model = joblib.load("logistic_model.pkl")

# Definir el esquema de entrada
class PredictionRequest(BaseModel):
    OPERA_Latin_American_Wings: float
    MES_7: float
    MES_10: float
    OPERA_Grupo_LATAM: float
    MES_12: float
    TIPOVUELO_1: float
    MES_4: float
    MES_11: float
    OPERA_Sky_Airline: float
    OPERA_Copa_Air: float

# Endpoint de salud
@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}

# Endpoint de predicción con XGBoost
@app.post("/predict_xgb", status_code=200)
async def post_predict_xgb(request: PredictionRequest) -> dict:
    data = pd.DataFrame([request.dict().values()], columns=request.dict().keys())
    prediction = xgb_model.predict(data)
    return {"prediction": int(prediction[0])}

# Endpoint de predicción con Logistic Regression
@app.post("/predict_logistic", status_code=200)
async def post_predict_logistic(request: PredictionRequest) -> dict:
    data = pd.DataFrame([request.dict().values()], columns=request.dict().keys())
    prediction = logistic_model.predict(data)
    return {"prediction": int(prediction[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

