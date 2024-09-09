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
def load_and_prepare_data(filepath):
    data = pd.read_csv(filepath)
    training_data = shuffle(data[['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM', 'delay']], random_state=111)
    features = pd.concat([
        pd.get_dummies(data['OPERA'], prefix='OPERA'),
        pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
        pd.get_dummies(data['MES'], prefix='MES')
    ], axis=1)
    target = data['delay']
    return train_test_split(features, target, test_size=0.33, random_state=42)

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

# Modelos y evaluación
def train_and_evaluate_models(x_train, y_train, x_test, y_test):
    # Modelo XGBoost con balance
    scale = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    xgb_model_2 = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight=scale)
    xgb_model_2.fit(x_train, y_train)
    xgboost_y_preds_2 = xgb_model_2.predict(x_test)
    print("XGBoost with Balance Classification Report:")
    print(classification_report(y_test, xgboost_y_preds_2))
    xgb_cm_2 = confusion_matrix(y_test, xgboost_y_preds_2)
    sns.heatmap(xgb_cm_2, annot=True, fmt='d', cmap='Blues')
    plt.title('XGBoost with Balance Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    plot_importance(xgb_model_2)
    plt.title('Feature Importance - XGBoost with Balance')
    plt.show()

    # Modelo de Regresión Logística con balance
    n_y0 = len(y_train[y_train == 0])
    n_y1 = len(y_train[y_train == 1])
    reg_model_2 = LogisticRegression(class_weight={1: n_y0/len(y_train), 0: n_y1/len(y_train)})
    reg_model_2.fit(x_train, y_train)
    reg_y_preds_2 = reg_model_2.predict(x_test)
    print("Logistic Regression with Balance Classification Report:")
    print(classification_report(y_test, reg_y_preds_2))
    reg_cm_2 = confusion_matrix(y_test, reg_y_preds_2)
    sns.heatmap(reg_cm_2, annot=True, fmt='d', cmap='Blues')
    plt.title('Logistic Regression with Balance Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Visualización de la importancia de características para Regresión Logística
    coefficients = pd.DataFrame({"Feature": x_train.columns, "Coefficient": reg_model_2.coef_[0]})
    coefficients = coefficients.sort_values(by="Coefficient", ascending=False)
    plt.figure(figsize=(10, 8))
    sns.barplot(x="Coefficient", y="Feature", data=coefficients)
    plt.title('Feature Importance - Logistic Regression with Balance')
    plt.show()

# Función para determinar si hay retraso
def is_delay(departure, arrival):
    min_diff = (arrival - departure).total_seconds() / 60
    return 1 if min_diff > 15 else 0

# Ejemplo de uso de la función is_delay
def example_is_delay():
    departure_time = datetime.strptime('2023-09-09 14:30:00', '%Y-%m-%d %H:%M:%S')
    arrival_time = datetime.strptime('2023-09-09 14:50:00', '%Y-%m-%d %H:%M:%S')
    delay_status = is_delay(departure_time, arrival_time)
    print(f"Delay status: {delay_status}")

# Función principal para ejecutar todo el proceso.
def main():
    x_train, x_test, y_train, y_test = load_and_prepare_data('data/data.csv')
    plot_delay_rate(data, 'MES', 'Delay Rate by Month', 10)
    plot_delay_rate(data, 'DIANOM', 'Delay Rate by Day', 7)
    plot_delay_rate(data, 'high_season', 'Delay Rate by Season', 6)
    plot_delay_rate(data, 'TIPOVUELO', 'Delay Rate by Flight Type', 7)
    train_and_evaluate_models(x_train, y_train, x_test, y_test)
    example_is_delay()

if __name__ == "__main__":
    main()

