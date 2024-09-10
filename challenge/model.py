import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

# Cargar datos
df = pd.read_csv('.../data/data.csv')
print(df.head())

# Generación de características
def get_day_period(date):
    morning_min = datetime.strptime("05:00", "%H:%M").time()
    morning_max = datetime.strptime("11:59", "%H:%M").time()
    afternoon_min = datetime.strptime("12:00", "%H:%M").time()
    afternoon_max = datetime.strptime("18:59", "%H:%M").time()
    evening_min = datetime.strptime("19:00", "%H:%M").time()
    evening_max = datetime.strptime("23:59", "%H:%M").time()
    night_min = datetime.strptime("00:00", "%H:%M").time()
    night_max = datetime.strptime("04:59", "%H:%M").time()

    if morning_min <= date.time() <= morning_max:
        return 'morning'
    elif afternoon_min <= date.time() <= afternoon_max:
        return 'afternoon'
    elif evening_min <= date.time() <= evening_max:
        return 'evening'
    else:
        return 'night'

df['period_day'] = df['Fecha-I'].apply(lambda x: get_day_period(datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')))

def is_high_season(fecha):
    fecha_año = int(fecha.split('-')[0])
    fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S.%f')
    
    rangel_min = datetime.strptime('15-Dec', '%d-%b').replace(year=fecha_año)
    rangel_max = datetime.strptime('31-Dec', '%d-%b').replace(year=fecha_año)
    range2_min = datetime.strptime('01-Jan', '%d-%b').replace(year=fecha_año)
    range2_max = datetime.strptime('03-Mar', '%d-%b').replace(year=fecha_año)
    range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year=fecha_año)
    range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year=fecha_año)
    range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year=fecha_año)
    range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year=fecha_año)

    if ((rangel_min <= fecha <= rangel_max) or
        (range2_min <= fecha <= range2_max) or
        (range3_min <= fecha <= range3_max) or
        (range4_min <= fecha <= range4_max)):
        return 1
    else:
        return 0

df['high_season'] = df['Fecha-I'].apply(lambda x: is_high_season(x))

def get_min_diff(data):
    fecha_o = datetime.strptime(data["Fecha-O"], '%Y-%m-%d %H:%M:%S')
    fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
    min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
    return min_diff

df['min_diff'] = df.apply(get_min_diff, axis=1)

threshold_in_minutes = 15
df['delay'] = np.where(df['min_diff'] > threshold_in_minutes, 1, 0)

# Selección de características
top_features = ['OPERA_Latin American Wings', 'MES_7', 'MES_10', 'OPERA_Grupo LATAM', 'MES_12', 'TIPOVUELO_1', 'MES_4', 'MES_11', 'OPERA_Sky Airline', 'OPERA_Copa Air']

# 6.a Data Split
X_train2, X_test2, y_train2, y_test2 = train_test_split(df[top_features], df['delay'], test_size=0.33, random_state=42)

# Balanceo de clases
n_y0 = len(y_train2[y_train2 == 0])
n_y1 = len(y_train2[y_train2 == 1])
scale = n_y0 / n_y1

# 6.b Model Selection

## 6.b.i XGBoost with Feature Importance and with Balance
xgb_model2 = xgb.XGBClassifier(scale_pos_weight=scale, learning_rate=0.01, n_estimators=2000, max_depth=4)
xgb_model2.fit(X_train2, y_train2)
y_pred5 = xgb_model2.predict(X_test2)

## 6.b.ii XGBoost with Feature Importance but without Balance
xgb_model3 = xgb.XGBClassifier(learning_rate=0.01)
xgb_model3.fit(X_train2, y_train2)
y_pred6 = xgb_model3.predict(X_test2)

## 6.b.iii Logistic Regression with Feature Importance and with Balance
reg_model_1 = LogisticRegression(class_weight='balanced')
reg_model_1.fit(X_train2, y_train2)
y_pred7 = reg_model_1.predict(X_test2)

# Evaluación del modelo
print(confusion_matrix(y_test2, y_pred5))
print(classification_report(y_test2, y_pred5))

# Guardar el modelo XGBoost
joblib.dump(xgb_model2, "xgb_model.pkl")

# Guardar el modelo Logistic Regression
joblib.dump(reg_model_1, "logistic_model.pkl")
