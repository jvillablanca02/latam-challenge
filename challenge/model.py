
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance
from sklearn.utils.class_weight import compute_class_weight
from typing import Union, Tuple, List
from pathlib import Path
import logging
import json

warnings.filterwarnings('ignore')

# Cargar Datos
data = pd.read_csv('./data/data.csv')

# Generación de Características
def get_period_day(date):
    date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
    morning_min = datetime.strptime("05:00", '%H:%M').time()
    morning_max = datetime.strptime("11:59", '%H:%M').time()
    afternoon_min = datetime.strptime("12:00", '%H:%M').time()
    afternoon_max = datetime.strptime("18:59", '%H:%M').time()
    evening_min = datetime.strptime("19:00", '%H:%M').time()
    evening_max = datetime.strptime("23:59", '%H:%M').time()
    night_min = datetime.strptime("00:00", '%H:%M').time()
    night_max = datetime.strptime("04:59", '%H:%M').time()

    if morning_min <= date_time <= morning_max:
        return 'morning'
    elif afternoon_min <= date_time <= afternoon_max:
        return 'afternoon'
    elif evening_min <= date_time <= evening_max or night_min <= date_time <= night_max:
        return 'night'

data['period_day'] = data['Fecha-I'].apply(get_period_day)

def is_high_season(fecha):
    fecha_año = int(fecha.split('-')[0])
    fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
    range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year=fecha_año)
    range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year=fecha_año)
    range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year=fecha_año)
    range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year=fecha_año)
    range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year=fecha_año)
    range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year=fecha_año)
    range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year=fecha_año)
    range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year=fecha_año)

    if ((range1_min <= fecha <= range1_max) or
        (range2_min <= fecha <= range2_max) or
        (range3_min <= fecha <= range3_max) or
        (range4_min <= fecha <= range4_max)):
        return 1
    else:
        return 0

data['high_season'] = data['Fecha-I'].apply(is_high_season)

def get_min_diff(data):
    fecha_0 = datetime.strptime(data['Fecha_0'], '%Y-%m-%d %H:%M:%S')
    fecha_1 = datetime.strptime(data['Fecha_1'], '%Y-%m-%d %H:%M:%S')
    min_diff = (fecha_0 - fecha_1).total_seconds() / 60
    return min_diff

data['min_diff'] = data.apply(get_min_diff, axis=1)

threshold_in_minutes = 15
data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)

# Clase para manejar el modelo de retraso
class DelayModel:

    TOP_FEATURES = [
        "OPERA_Latin American Wings", 
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air"
    ]

    DATA_SPLITTING_RANDOM_STATE = 42
    TEST_SIZE = 0.33

    MODEL_RANDOM_STATE = 1
    LEARNING_RATE = 0.01

    DELAY_THRESHOLD_MINUTES = 15

    MODEL_FILE_NAME = Path("model.json")
    MODEL_PATH = Path("models")

    def __init__(self):
        self._model: Union[BaseEstimator, ClassifierMixin] = self._load_model(
            self.complete_model_path
        )
        self.target_column = None

    @property
    def complete_model_path(self) -> Path:
        return self.MODEL_PATH / self.MODEL_FILE_NAME

    def _get_minute_diff(self, data: pd.DataFrame) -> pd.Series:
        try:
            fecha_o = pd.to_datetime(data["Fecha-O"])
            fecha_i = pd.to_datetime(data["Fecha-I"])
            return (fecha_o - fecha_i).dt.total_seconds() / 60
        except (ValueError, KeyError) as e:
            raise ValueError("Datos de entrada o formato de fecha inválidos") from e

    def _create_one_hot_features(self, data: pd.DataFrame) -> pd.DataFrame:
        one_hot_features = pd.concat(
            [
                pd.get_dummies(data["OPERA"], prefix="OPERA"),
                pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
                pd.get_dummies(data["MES"], prefix="MES"),
            ],
            axis=1,
        )
        return one_hot_features

    def _create_delay_target(
        self, data: pd.DataFrame, target_column: str
    ) -> pd.DataFrame:
        data["min_diff"] = self._get_minute_diff(data)
        delay_target = np.where(data["min_diff"] > self.DELAY_THRESHOLD_MINUTES, 1, 0)
        return pd.DataFrame({target_column: delay_target}, index=data.index)

    def _load_model(self, model_path: Path) -> Union[BaseEstimator, ClassifierMixin]:
        logging.info(f"Cargando modelo desde {model_path}")

        try:
            if not model_path.is_file():
                raise FileNotFoundError(f"No se encontró ningún archivo en {model_path}")

            if model_path.suffix.lower() != ".json":
                logging.warning(
                    f"El archivo {model_path} no tiene una extensión .json. Intentando cargar de todas formas."
                )

            model = XGBClassifier()
            model.load_model(str(model_path))

            logging.info("Modelo cargado exitosamente")

            return model

        except FileNotFoundError as e:
            logging.error(f"Archivo del modelo no encontrado: {e}")
        except json.JSONDecodeError:
            logging.error(
                f"Error al decodificar JSON desde {model_path}. Asegúrate de que sea un archivo JSON válido."
            )
        except PermissionError:
            logging.error(f"Permiso denegado al intentar leer {model_path}")
        except Exception as e:
            logging.error(f"Error al cargar el modelo: {str(e)}")

    def _calculate_class_weights(self, target: pd.Series) -> dict:
        classes = np.unique(target)
        weights = compute_class_weight(
            class_weight="balanced", classes=classes, y=target
        )
        class_weights = dict(zip(classes, weights))
        print(class_weights)
        return class_weights

    def preprocess(
        self, data: pd.DataFrame, target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        target = None

        if target_column:
            self.target_column = target_column
            target = self._create_delay_target(data, target_column)

        features = self._create_one_hot_features(data)
        features = features.reindex(columns=self.TOP_FEATURES, fill_value=0)

        if target_column:
            return features, target
        else:
            return features

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=self.TEST_SIZE,
            random_state=self.DATA_SPLITTING_RANDOM_STATE,
        )

        target_series = target[self.target_column]
        class_weights = self._calculate_class_weights(target=target_series)
        scale_pos_weight = class_weights[1] / class_weights[0]
        print(scale_pos_weight)

        model = XGBClassifier(
            random_state=self.MODEL_RANDOM_STATE,
            learning_rate=self.LEARNING_RATE,
            scale_pos_weight=scale_pos_weight,
        )

        model.fit(x_train, y_train)

        logging.info("Entrenamiento finalizado, calculando métricas de prueba...")

        y_pred = model.predict(x_test)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        logging.info(f"Matriz de Confusión:\n{conf_matrix}")
        logging.info(f"Reporte de Clasificación:\n{class_report}")

        self.complete_model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(self.complete_model_path)
        self._model = model

       def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predice retrasos para nuevos vuelos.

        Args:
            features (pd.DataFrame): datos preprocesados.

        Returns:
            (List[int]): objetivos predichos.
        """
        if self._model is None:
            logging.warning("No se encontró el modelo.")
            self._load_model(self.complete_model_path)

        predictions = self._model.predict(features)
        return predictions.tolist()

