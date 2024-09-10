from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import joblib

app = FastAPI()

# Cargar los modelos previamente entrenados
xgb_model = joblib.load("xgb_model.pkl")
logistic_model = joblib.load("logistic_model.pkl")

class FlightData(BaseModel):
    OPERA: str
    MES: int
    TIPOVUELO: str
    SIGLADES: str
    DIANOM: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict_delay(data: FlightData):
    try:
        # Convertir los datos de entrada en un DataFrame
        input_data = pd.DataFrame([data.dict()])

        # Preprocesar los datos de entrada de la misma manera que se hizo durante el entrenamiento
        input_data = pd.concat([
            pd.get_dummies(input_data['OPERA'], prefix='OPERA'),
            pd.get_dummies(input_data['TIPOVUELO'], prefix='TIPOVUELO'),
            pd.get_dummies(input_data['MES'], prefix='MES')
        ], axis=1)

        # Asegurarse de que las columnas coincidan con las del modelo entrenado
        missing_cols = set(xgb_model.feature_names) - set(input_data.columns)
        for col in missing_cols:
            input_data[col] = 0
        input_data = input_data[xgb_model.feature_names]

        # Realizar la predicción con ambos modelos
        xgb_prediction = xgb_model.predict(input_data)
        logistic_prediction = logistic_model.predict(input_data)

        return {
            "xgb_prediction": int(xgb_prediction[0]),
            "logistic_prediction": int(logistic_prediction[0])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
