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
from xgboost import plot_importance

warnings.filterwarnings('ignore')

# 0. Cargar Datos
data = pd.read_csv('./data/data.csv')
data.info()

# 1. Análisis de Datos: Primera Vista
# Distribución de vuelos por aerolínea
flights_by_airline = data['OPERA'].value_counts()
plt.figure(figsize=(10, 2))
sns.set(style="darkgrid")
sns.barplot(flights_by_airline.index, flights_by_airline.values, alpha=0.8)
plt.title("Vuelos por Aerolínea")
plt.ylabel("Vuelos", fontsize=12)
plt.xlabel("Aerolínea", fontsize=12)
plt.xticks(rotation=90)
plt.show()

# Distribución de vuelos por día
flights_by_day = data['DIA'].value_counts()
plt.figure(figsize=(10, 2))
sns.set(style="darkgrid")
sns.barplot(flights_by_day.index, flights_by_day.values, color="lightblue", alpha=0.8)
plt.title("Vuelos por Día")
plt.ylabel("Vuelos", fontsize=12)
plt.xlabel("Día", fontsize=12)
plt.xticks(rotation=90)
plt.show()

# Distribución de vuelos por mes
flights_by_month = data['MES'].value_counts()
plt.figure(figsize=(10, 2))
sns.set(style="darkgrid")
sns.barplot(flights_by_month.index, flights_by_month.values, color="lightblue", alpha=0.8)
plt.title("Vuelos por Mes")
plt.ylabel("Vuelos", fontsize=12)
plt.xlabel("Mes", fontsize=12)
plt.xticks(rotation=90)
plt.show()

# Distribución de vuelos por día de la semana
flights_by_day_in_week = data['DIANOM'].value_counts()
days = [
    flights_by_day_in_week.index[2],
    flights_by_day_in_week.index[5],
    flights_by_day_in_week.index[4],
    flights_by_day_in_week.index[1],
    flights_by_day_in_week.index[0],
    flights_by_day_in_week.index[6],
    flights_by_day_in_week.index[3]
]
values_by_day = [
    flights_by_day_in_week.values[2],
    flights_by_day_in_week.values[5],
    flights_by_day_in_week.values[4],
    flights_by_day_in_week.values[1],
    flights_by_day_in_week.values[0],
    flights_by_day_in_week.values[6],
    flights_by_day_in_week.values[3]
]
plt.figure(figsize=(10, 2))
sns.set(style="darkgrid")
sns.barplot(days, values_by_day, color="lightblue", alpha=0.8)
plt.title("Vuelos por Día de la Semana")
plt.ylabel("Vuelos", fontsize=12)
plt.xlabel("Día de la Semana", fontsize=12)
plt.xticks(rotation=90)
plt.show()

# 2. Generación de Características
# 2.a. Periodo del Día
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

# 2.b. Temporada Alta
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

# 2.c. Diferencia en Minutos
def get_min_diff(data):
    fecha_0 = datetime.strptime(data['Fecha_0'], '%Y-%m-%d %H:%M:%S')
    fecha_1 = datetime.strptime(data['Fecha_1'], '%Y-%m-%d %H:%M:%S')
    min_diff = (fecha_0 - fecha_1).total_seconds() / 60
    return min_diff

data['min_diff'] = data.apply(get_min_diff, axis=1)

# 2.d. Retraso
threshold_in_minutes = 15
data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)

# 3. Análisis de Datos: Segunda Vista
# Tasa de retraso por columna
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
    for name, total_count in total.items():
        if name in delays:
            rates[name] = round((delays[name] / total_count) * 100, 2)
        else:
            rates[name] = 0

    return pd.DataFrame.from_dict(rates, orient='index', columns=['Tasa (%)'])

# Tasa de retraso por destino
destination_rate = get_rate_from_column(data, 'SIGLADES')
destination_rate_values = data['SIGLADES'].value_counts().index
plt.figure(figsize=(20, 5))
sns.set(style="darkgrid")
sns.barplot(destination_rate_values, destination_rate['Tasa (%)'], alpha=0.75)
plt.title('Tasa de Retraso por Destino')
plt.ylabel('Tasa de Retraso [%]', fontsize=12)
plt.xlabel('Destino', fontsize=12)
plt.xticks(rotation=90)
plt.show()

# Tasa de retraso por aerolínea
airlines_rate = get_rate_from_column(data, 'OPERA')
airlines_rate_values = data['OPERA'].value_counts().index
plt.figure(figsize=(20, 5))
sns.set(style="darkgrid")
sns.barplot(airlines_rate_values, airlines_rate['Tasa (%)'], alpha=0.75)
plt.title('Tasa de Retraso por Aerolínea')
plt.ylabel('Tasa de Retraso [%]', fontsize=12)
plt.xlabel('Aerolínea', fontsize=12)
plt.xticks(rotation=90)
plt.show()

# Tasa de retraso por mes
month_rate = get_rate_from_column(data, 'MES')
month_rate_values = data['MES'].value_counts().index
plt.figure(figsize=(20, 5))
sns.set(style="darkgrid")
sns.barplot(month_rate_values, month_rate['Tasa (%)'], color='blue', alpha=0.75)
plt.title('Tasa de Retraso por Mes')
plt.ylabel('Tasa de Retraso [%]', fontsize=12)
plt.xlabel('Mes', fontsize=12)
plt.xticks(rotation=90)
plt.ylim(0, 10)
plt.show()

# Tasa de retraso por día
days_rate = get_rate_from_column(data, 'DIANOM')
days_rate_value = data['DIANOM'].value_counts().index
plt.figure(figsize=(20, 5))
sns.set(style="darkgrid")
sns.barplot(days_rate_value, days_rate['Tasa (%)'], color='blue', alpha=0.75)
plt.title('Tasa de Retraso por Día')
plt.ylabel('Tasa de Retraso [%]', fontsize=12)
plt.xlabel('Día', fontsize=12)
plt.xticks(rotation=90)
plt.ylim(0, 10)
plt.show()

# 4. Entrenamiento
# 4.a. División de Datos (Entrenamiento y Validación)
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report

# Asegúrate de que los datos estén bien mezclados
training_data = shuffle(data[['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM', 'delay']], random_state=111)

# Crear variables dummy para las características categóricas
features = pd.concat([
    pd.get_dummies(data['OPERA'], prefix='OPERA'),
    pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
    pd.get_dummies(data['MES'], prefix='MES')
], axis=1)

# Definir el objetivo
target = data['delay']

# Dividir los datos en conjuntos de entrenamiento y validación
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)

# Verificar la forma de los conjuntos de datos
print(f"train shape: {x_train.shape} | test shape: {x_test.shape}")

# Verificar la distribución de la variable objetivo en los conjuntos de entrenamiento y validación
print(y_train.value_counts(normalize=True) * 100)
print(y_test.value_counts(normalize=True) * 100)

# 4.b. Selección de Modelos
# 4.b.i. XGBoost
import xgboost as xgb

# Crear y entrenar el modelo XGBoost
xgb_model = xgb.XGBClassifier(random_state=1, learning_rate=0.01)
xgb_model.fit(x_train, y_train)

# Guardar el modelo entrenado en un archivo .pkl
joblib.dump(xgb_model, "challenge/finalized_model.pkl")

# Realizar predicciones
xgboost_y_preds = xgb_model.predict(x_test)
xgboost_y_preds = [1 if y_pred > 0.5 else 0 for y_pred in xgboost_y_preds]

# Evaluar el modelo
print(confusion_matrix(y_test, xgboost_y_preds))
print(classification_report(y_test, xgboost_y_preds))

# 4.b.ii. Regresión Logística
from sklearn.linear_model import LogisticRegression

# Crear y entrenar el modelo de Regresión Logística
reg_model = LogisticRegression()
reg_model.fit(x_train, y_train)

# Realizar predicciones
reg_y_preds = reg_model.predict(x_test)

# Evaluar el modelo
print(confusion_matrix(y_test, reg_y_preds))
print(classification_report(y_test, reg_y_preds))

# 5. Análisis de Datos: Tercera Vista
# Importancia de las Características
plt.figure(figsize=(10, 5))
plot_importance(xgb_model)
plt.show()

# Seleccionar las 10 características más importantes
top_10_features = [
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

# Balance de Datos
n_y0 = len(y_train[y_train == 0])
n_y1 = len(y_train[y_train == 1])
scale = n_y0 / n_y1
print(scale)

# 6. Entrenamiento con Mejora
# 6.a. División de Datos
x_train2, x_test2, y_train2, y_test2 = train_test_split(features[top_10_features], target, test_size=0.33, random_state=42)

# 6.b. Selección de Modelos
# 6.b.i. XGBoost con Importancia de Características y Balance
xgb_model_2 = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight=scale)
xgb_model_2.fit(x_train2, y_train2)
xgboost_y_preds_2 = xgb_model_2.predict(x_test2)
print(confusion_matrix(y_test2, xgboost_y_preds_2))
print(classification_report(y_test2, xgboost_y_preds_2))

# 6.b.ii. XGBoost con Importancia de Características pero sin Balance
xgb_model_3 = xgb.XGBClassifier(random_state=1, learning_rate=0.01)
xgb_model_3.fit(x_train2, y_train2)
xgboost_y_preds_3 = xgb_model_3.predict(x_test2)
print(confusion_matrix(y_test2, xgboost_y_preds_3))
print(classification_report(y_test2, xgboost_y_preds_3))

# 6.b.iii. Regresión Logística con Importancia de Características y Balance
reg_model_2 = LogisticRegression(class_weight={1: n_y0/len(y_train), 0: n_y1/len(y_train)})
reg_model_2.fit(x_train2, y_train2)
reg_y_preds_2 = reg_model_2.predict(x_test2)
print(confusion_matrix(y_test2, reg_y_preds_2))
print(classification_report(y_test2, reg_y_preds_2))

# 6.b.iv. Regresión Logística con Importancia de Características pero sin Balance
reg_model_3 = LogisticRegression()
reg_model_3.fit(x_train2, y_train2)
reg_y_preds_3 = reg_model_3.predict(x_test2)
print(confusion_matrix(y_test2, reg_y_preds_3))
print(classification_report(y_test2, reg_y_preds_3))

# 7. Conclusiones de Ciencia de Datos
print("""
Conclusiones de Ciencia de Datos:

1. No hay una diferencia notable en los resultados entre XGBoost y Regresión Logística.
2. No disminuye el rendimiento del modelo al reducir las características a las 10 más importantes.
3. Mejora el rendimiento del modelo al balancear las clases, ya que aumenta el recall de la clase "1".

Con esto, el modelo a ser productivo debe ser el que se entrena con las 10 características principales y balanceo de clases.
""")

