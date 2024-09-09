  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
from xgboost import plot_importance
from sklearn.linear_model import LogisticRegression
from datetime import datetime

# Función para calcular la tasa de retraso por columna
def get_rate_from_column(data, column):
    delays = data[data['delay'] == 1][column].value_counts()
    total = data[column].value_counts()
    rates = (delays / total).fillna(0) * 100
    return rates.reset_index().rename(columns={column: 'Tasa (%)', 'index': column})

# Cargar y preparar los datos
data = pd.read_csv('data/data.csv')  
training_data = shuffle(data[['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM', 'delay']], random_state=111)
features = pd.concat([
    pd.get_dummies(data['OPERA'], prefix='OPERA'),
    pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
    pd.get_dummies(data['MES'], prefix='MES')
], axis=1)
target = data['delay']
x_train2, x_test2, y_train2, y_test2 = train_test_split(features, target, test_size=0.33, random_state=42)

# Visualización de tasas de retraso
def plot_delay_rate(data, column, title, ylim):
    rate = get_rate_from_column(data, column)
    plt.figure(figsize=(10, 5))
    sns.set(style="darkgrid")
    bar_plot = sns.barplot(x=column, y='Tasa (%)', data=rate, alpha=0.75)
    plt.title(title, fontsize=14)
    plt.ylabel('Delay Rate [%]', fontsize=12)
    plt.xlabel(column, fontsize=12)
    plt.ylim(0, ylim)
    for index, value in enumerate(rate['Tasa (%)']):
        bar_plot.text(index, value, f'{value:.2f}', color='black', ha="center")
    plt.show()

plot_delay_rate(data, 'MES', 'Delay Rate by Month', 10)
plot_delay_rate(data, 'DIANOM', 'Delay Rate by Day', 7)
plot_delay_rate(data, 'high_season', 'Delay Rate by Season', 6)
plot_delay_rate(data, 'TIPOVUELO', 'Delay Rate by Flight Type', 7)

# Modelo XGBoost con balance
scale = len(y_train2[y_train2 == 0]) / len(y_train2[y_train2 == 1])
xgb_model_2 = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight=scale)
xgb_model_2.fit(x_train2, y_train2)
xgboost_y_preds_2 = xgb_model_2.predict(x_test2)

# Reporte de clasificación y matriz de confusión para XGBoost con balance
print("XGBoost with Balance Classification Report:")
print(classification_report(y_test2, xgboost_y_preds_2))
xgb_cm_2 = confusion_matrix(y_test2, xgboost_y_preds_2)
sns.heatmap(xgb_cm_2, annot=True, fmt='d', cmap='Blues')
plt.title('XGBoost with Balance Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Importancia de características para XGBoost con balance
plot_importance(xgb_model_2)
plt.title('Feature Importance - XGBoost with Balance')
plt.show()

# Modelo XGBoost sin balance
xgb_model_3 = xgb.XGBClassifier(random_state=1, learning_rate=0.01)
xgb_model_3.fit(x_train2, y_train2)
xgboost_y_preds_3 = xgb_model_3.predict(x_test2)

# Reporte de clasificación y matriz de confusión para XGBoost sin balance
print("XGBoost without Balance Classification Report:")
print(classification_report(y_test2, xgboost_y_preds_3))
xgb_cm_3 = confusion_matrix(y_test2, xgboost_y_preds_3)
sns.heatmap(xgb_cm_3, annot=True, fmt='d', cmap='Blues')
plt.title('XGBoost without Balance Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Modelo de Regresión Logística con balance
n_y0 = len(y_train2[y_train2 == 0])
n_y1 = len(y_train2[y_train2 == 1])
reg_model_2 = LogisticRegression(class_weight={1: n_y0/len(y_train2), 0: n_y1/len(y_train2)})
reg_model_2.fit(x_train2, y_train2)
reg_y_preds_2 = reg_model_2.predict(x_test2)

# Reporte de clasificación y matriz de confusión para Regresión Logística con balance
print("Logistic Regression with Balance Classification Report:")
print(classification_report(y_test2, reg_y_preds_2))
reg_cm_2 = confusion_matrix(y_test2, reg_y_preds_2)
sns.heatmap(reg_cm_2, annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression with Balance Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Modelo de Regresión Logística sin balance
reg_model_3 = LogisticRegression()
reg_model_3.fit(x_train2, y_train2)
reg_y_preds_3 = reg_model_3.predict(x_test2)

# Reporte de clasificación y matriz de confusión para Regresión Logística sin balance
print("Logistic Regression without Balance Classification Report:")
print(classification_report(y_test2, reg_y_preds_3))
reg_cm_3 = confusion_matrix(y_test2, reg_y_preds_3)
sns.heatmap(reg_cm_3, annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression without Balance Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Función para determinar si hay retraso
def is_delay(departure, arrival):
    min_diff = (arrival - departure).total_seconds() / 60
    return 1 if min_diff > 15 else 0

# Ejemplo de uso de la función is_delay
departure_time = datetime.strptime('2023-09-09 14:30:00', '%Y-%m-%d %H:%M:%S')
arrival_time = datetime.strptime('2023-09-09 14:50:00', '%Y-%m-%d %H:%M:%S')
delay_status = is_delay(departure_time, arrival_time)
print(f"Delay status: {delay_status}")

# Visualización de la importancia de características para Regresión Logística
coefficients = pd.DataFrame({"Feature": features.columns, "Coefficient": reg_model_3.coef_[0]})
coefficients = coefficients.sort_values(by="Coefficient", ascending=False)
plt.figure(figsize=(10, 8))
sns.barplot(x="Coefficient", y="Feature", data=coefficients)
plt.title('Feature Importance - Logistic Regression without Balance')
plt.show()
 
 
