  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
from typing import Tuple, Union, List
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

class DelayModel:

    def __init__(self):
        self._model = RandomForestClassifier()  # El modelo debe guardarse en este atributo.

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepara los datos en bruto para entrenamiento o predicción.

        Args:
            data (pd.DataFrame): datos en bruto.
            target_column (str, opcional): si se establece, se devuelve el objetivo.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: características y objetivo.
            o
            pd.DataFrame: características.
        """
        # Ejemplo de preprocesamiento: llenar valores faltantes y codificar variables categóricas
        data = data.fillna(method='ffill')
        if target_column:
            features = data.drop(columns=[target_column])
            target = data[target_column]
            return features, target
        else:
            return data

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Ajusta el modelo con datos preprocesados.

        Args:
            features (pd.DataFrame): datos preprocesados.
            target (pd.DataFrame): objetivo.
        """
        self._model.fit(features, target)

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predice retrasos para nuevos vuelos.

        Args:
            features (pd.DataFrame): datos preprocesados.
        
        Returns:
            List[int]: objetivos predichos.
        """
        predictions = self._model.predict(features)
        return predictions.tolist()

# Ejemplo de uso
if __name__ == "__main__":
    # Cargar datos
    data = pd.read_csv('../data/data.csv')
    
    # Mostrar información de los datos
    data.info()
    
    # Generar característica de período del día
    def get_period_of_day(date: str) -> str:
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
            return 'mañana'
        elif afternoon_min <= date_time <= afternoon_max:
            return 'tarde'
        elif evening_min <= date_time <= evening_max or night_min <= date_time <= night_max:
            return 'noche'

    data['period_day'] = data['Fecha-I'].apply(get_period_of_day)
    
    # Generar característica de temporada alta
    def is_high_season(fecha: str) -> int:
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

        if ((fecha >= range1_min and fecha <= range1_max) or
            (fecha >= range2_min and fecha <= range2_max) or
            (fecha >= range3_min and fecha <= range3_max) or
            (fecha >= range4_min and fecha <= range4_max)):
            return 1
        else:
            return 0

    data['high_season'] = data['Fecha-I'].apply(is_high_season)
    
    # Generar características de min_diff y delay
    def get_min_diff(data):
        fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
        return min_diff

    data['min_diff'] = data.apply(get_min_diff, axis=1)
    
    threshold_in_minutes = 15
    data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)
    
    # Inicializar modelo
    model = DelayModel()
    
    # Preprocesar datos
    features, target = model.preprocess(data, target_column='delay')
    
    # División de los datos en conjuntos de entrenamiento y validación
    features = pd.concat([pd.get_dummies(data['OPERA'], prefix='OPERA'),
                          pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
                          pd.get_dummies(data['MES'], prefix='MES')], axis=1)
    target = data['delay']
    
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)
    
    print(f"train shape: {x_train.shape} | test shape: {x_test.shape}")
    print(y_train.value_counts(normalize=True) * 100)
    print(y_test.value_counts(normalize=True) * 100)
    
    # Ajustar modelo
    model.fit(x_train, y_train)
    
    # Predecir
    predictions = model.predict(x_test)
    
    # Evaluar
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model accuracy: {accuracy}")
    
    # Análisis de Datos: Primera Vista
    # Distribución por Día
    flights_by_day = data['DIA'].value_counts()
    plt.figure(figsize=(18, 2))
    sns.set(style="darkgrid")
    sns.barplot(x=flights_by_day.index, y=flights_by_day.values, color="lightblue", alpha=0.3)
    plt.title("Flights by Day")
    plt.ylabel("Flights", fontsize=12)
    plt.xlabel("Day", fontsize=12)
    plt.xticks(rotation=90)
    plt.show()

    # Distribución por Mes
    flights_by_month = data['MES'].value_counts()
    plt.figure(figsize=(18, 2))
    sns.set(style="darkgrid")
    sns.barplot(x=flights_by_month.index, y=flights_by_month.values, color="lightblue", alpha=0.8)
    plt.title("Flights by Month")
    plt.ylabel("Flights", fontsize=12)
    plt.xlabel("Month", fontsize=12)
    plt.xticks(rotation=90)
    plt.show()
    # Distribución por Día de la Semana
    flights_by_day_in_week = data['DIANOM'].value_counts()
    days = flights_by_day_in_week.index
    values_by_day = flights_by_day_in_week.values
    plt.figure(figsize=(18, 2))
    sns.set(style="darkgrid")
    sns.barplot(x=days, y=values_by_day, color="lightblue", alpha=0.8)
    plt.title("Flights by Day in Week")
    plt.ylabel("Flights", fontsize=12)
    plt.xlabel("Day in Week", fontsize=12)
    plt.xticks(rotation=90)
    plt.show()

    # Distribución por Tipo de Vuelo
    flights_by_type = data['TIPOVUELO'].value_counts()
    plt.figure(figsize=(10, 2))
    sns.set(style="darkgrid")
    sns.barplot(x=flights_by_type.index, y=flights_by_type.values, alpha=0.9)
    plt.title("Flights by Type")
    plt.ylabel("Flights", fontsize=12)
    plt.xlabel("Type", fontsize=12)
    plt.show()

    # Distribución por Destino
    flight_by_destination = data['SIGLADES'].value_counts()
    plt.figure(figsize=(10, 2))
    sns.set(style="darkgrid")
    sns.barplot(x=flight_by_destination.index, y=flight_by_destination.values, color="lightblue", alpha=0.8)
    plt.title("Flight by Destination")
    plt.ylabel("Flights", fontsize=12)
    plt.xlabel("Destination", fontsize=12)
    plt.xticks(rotation=90)
    plt.show()

    # Análisis de Datos: Segunda Vista
    def get_rate_from_column(data, column):
        delays = {}
        for _, row in data.iterrows():
            if row['delay'] == 1:
                if row[column] not in delays:
                    delays[row[column]] = 1
                else:
                    delays[row[column]] += 1
        total = data[column].value_counts().to_dict()
        rates = {}
        for name, total in total.items():
            if name in delays:
                rates[name] = round((delays[name] / total) * 100, 2)
            else:
                rates[name] = 0
        return pd.DataFrame.from_dict(data=rates, orient='index', columns=['Tasa (%)'])

    # Tasa de retraso por destino
    destination_rate = get_rate_from_column(data, 'SIGLADES')
    destination_rate_values = data['SIGLADES'].value_counts().index

    plt.figure(figsize=(20, 5))
    sns.set(style="darkgrid")
    sns.barplot(destination_rate_values, destination_rate['Tasa (%)'], alpha=0.75)
    plt.title("Delay Rate by Destination")
    plt.ylabel("Delay Rate (%)", fontsize=12)
    plt.xlabel("Destination", fontsize=12)
    plt.xticks(rotation=90)
    plt.show()

    # Tasa de retraso por aerolínea
    airlines_rate = get_rate_from_column(data, 'OPERA')
    airlines_rate_values = data['OPERA'].value_counts().index

    plt.figure(figsize=(20, 5))
    sns.set(style="darkgrid")
    sns.barplot(airlines_rate_values, airlines_rate['Tasa (%)'], alpha=0.75)
    plt.title("Delay Rate by Airline")
    plt.ylabel("Delay Rate (%)", fontsize=12)
    plt.xlabel("Airline", fontsize=12)
    plt.xticks(rotation=90)
    plt.show()

    # Tasa de retraso por mes
    month_rate = get_rate_from_column(data, 'MES')
    month_rate_values = data['MES'].value_counts().index

    plt.figure(figsize=(20, 5))
    sns.set(style="darkgrid")
    sns.barplot(month_rate_values, month_rate['Tasa (%)'], color='blue', alpha=0.75)
    plt.title("Delay Rate by Month")
    plt.ylabel("Delay Rate (%)", fontsize=12)
    plt.xlabel("Month", fontsize=12)
    plt.xticks(rotation=90)
    plt.ylim(0, 10)
    plt.show()

    # Tasa de retraso por día
    days_rate = get_rate_from_column(data, 'DIANOM')
    days_rate_values = data['DIANOM'].value_counts().index

    plt.figure(figsize=(20, 5))
    sns.set(style="darkgrid")
    sns.barplot(days_rate_values, days_rate['Tasa (%)'], color='blue', alpha=0.75)
    plt.title("Delay Rate by Day")
    plt.ylabel("Delay Rate (%)", fontsize=12)
    plt.xlabel("Days", fontsize=12)
    plt.xticks(rotation=90)
    plt.ylim(0, 7)
    plt.show()

    # Tasa de retraso por temporada alta
    high_season_rate = get_rate_from_column(data, 'high_season')
    high_season_rate_values = data['high_season'].value_counts().index

    plt.figure(figsize=(5, 2))
    sns.set(style="darkgrid")
    sns.barplot(['no', 'yes'], high_season_rate['Tasa (%)'])
    plt.title("Delay Rate by Season")
    plt.ylabel("Delay Rate (%)", fontsize=12)
    plt.xlabel("High Season", fontsize=12)
    plt.xticks(rotation=90)
    plt.ylim(0, 7)
    plt.show()

    # Tasa de retraso por tipo de vuelo
    flight_type_rate = get_rate_from_column(data, 'TIPOVUELO')
    flight_type_rate_values = data['TIPOVUELO'].value_counts().index

    plt.figure(figsize=(5, 2))
    sns.set(style="darkgrid")
    sns.barplot(flight_type_rate_values, flight_type_rate['Tasa (%)'])
    plt.title("Delay Rate by Flight Type")
    plt.ylabel("Delay Rate (%)", fontsize=12)
    plt.xlabel("Flight Type", fontsize=12)
    plt.ylim(0, 7)
    plt.show()

    # Tasa de retraso por período del día
    period_day_rate = get_rate_from_column(data, 'period_day')
    period_day_rate_values = data['period_day'].value_counts().index

    plt.figure(figsize=(5, 2))
    sns.set(style="darkgrid")
    sns.barplot(period_day_rate_values, period_day_rate['Tasa (%)'])
    plt.title("Delay Rate by Period of Day")
    plt.ylabel("Delay Rate (%)", fontsize=12)
    plt.xlabel("Period", fontsize=12)
    plt.ylim(0, 7)
    plt.show()  
