import logging
from enum import Enum
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, conint

from challenge.model import DelayModel

app = FastAPI()

# Inicializar el modelo a nivel de módulo, pero solo cuando se necesite por primera vez
model = None

def get_model():
    global model
    if model is None:
        model = DelayModel()
    return model

class FlightType(str, Enum):
    NATIONAL = "N"
    INTERNATIONAL = "I"

class Airline(str, Enum):
    AMERICAN_AIRLINES = "American Airlines"
    AIR_CANADA = "Air Canada"
    AIR_FRANCE = "Air France"
    AEROMEXICO = "Aeromexico"
    AEROLINEAS_ARGENTINAS = "Aerolineas Argentinas"
    AUSTRAL = "Austral"
    AVIANCA = "Avianca"
    ALITALIA = "Alitalia"
    BRITISH_AIRWAYS = "British Airways"
    COPA_AIR = "Copa Air"
    DELTA_AIR = "Delta Air"
    GOL_TRANS = "Gol Trans"
    IBERIA = "Iberia"
    KLM = "K.L.M."
    QANTAS_AIRWAYS = "Qantas Airways"
    UNITED_AIRLINES = "United Airlines"
    GRUPO_LATAM = "Grupo LATAM"
    SKY_AIRLINE = "Sky Airline"
    LATIN_AMERICAN_WINGS = "Latin American Wings"
    PLUS_ULTRA_LINEAS_AEREAS = "Plus Ultra Lineas Aereas"
    JETSMART_SPA = "JetSmart SPA"
    OCEANAIR_LINHAS_AEREAS = "Oceanair Linhas Aereas"
    LACSA = "Lacsa"

class FlightData(BaseModel):
    OPERA: Airline
    TIPOVUELO: FlightType
    MES: conint(ge=1, le=12) = Field(..., description="Mes (1-12)")

class FlightsRequest(BaseModel):
    flights: List[FlightData]

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400, content={"detail": exc.errors(), "body": exc.body}
    )

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}

@app.post("/predict", status_code=200)
async def post_predict(request: FlightsRequest) -> dict:
    try:
        model = get_model()  # Obtener o inicializar el modelo
        flight_data = pd.DataFrame([flight.dict() for flight in request.flights])
        preprocessed_data = model.preprocess(flight_data)
        predictions = model.predict(preprocessed_data)
        return {"predict": predictions}
    except Exception as e:
        logging.exception("Ocurrió un error durante la predicción")
        raise HTTPException(status_code=500, detail="Ocurrió un error interno")

# Ejecutar la aplicación con Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

