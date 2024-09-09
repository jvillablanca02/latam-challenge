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

class DelayModel:

    def __init__(self):
        self._model = RandomForestClassifier()  # Model should be saved in this attribute.

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        # Example preprocessing: fill missing values and encode categorical variables
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
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        self._model.fit(features, target)

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            List[int]: predicted targets.
        """
        predictions = self._model.predict(features)
        return predictions.tolist()

# Example usage
if __name__ == "__main__":
    # Load data
    data = pd.read_csv('../data/data.csv')
    
    # Display data information
    data.info()
    
    # Generate period of day feature
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
    
    # Generate high season feature
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
    
    # Generate min_diff and delay features
    def get_min_diff(data):
        fecha_o = datetime.strptime(data['Fecha-0'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
        return min_diff

    data['min_diff'] = data.apply(get_min_diff, axis=1)
    
    threshold_in_minutes = 15
    data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)
    
    # Initialize model
    model = DelayModel()
    
    # Preprocess data
    features, target = model.preprocess(data, target_column='delay')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Fit model
    model.fit(X_train, y_train)
    
    # Predict
    predictions = model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model accuracy: {accuracy}")
    
    # Data Analysis: First Sight
    # Distribution by Day
    flights_by_day = data['day'].value_counts()
    plt.figure(figsize=(18, 2))
    sns.set(style="darkgrid")
    sns.barplot(x=flights_by_day.index, y=flights_by_day.values, color="lightblue", alpha=0.3)
    plt.title("Flights by Day")
    plt.ylabel("Flights", fontsize=12)
    plt.xlabel("Day", fontsize=12)
    plt.xticks(rotation=90)
    plt.show()

    # Distribution by Month
    flights_by_month = data['month'].value_counts()
    plt.figure(figsize=(18, 2))
    sns.set(style="darkgrid")
    sns.barplot(x=flights_by_month.index, y=flights_by_month.values, color="lightblue", alpha=0.8)
    plt.title("Flights by Month")
    plt.ylabel("Flights", fontsize=12)
    plt.xlabel("Month", fontsize=12)
    plt.xticks(rotation=90)
    plt.show()

    # Distribution by Day of the Week
    flights_by_day_in_week = data['day_of_week'].value_counts()
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

    # Distribution by Type of Flight
    flights_by_type = data['TIPOVUELO'].value_counts()
    plt.figure(figsize=(10, 2))
    sns.set(style="darkgrid")
    sns.barplot(x=flights_by_type.index, y=flights_by_type.values, alpha=0.9)
    plt.title("Flights by Type")
    plt.ylabel("Flights", fontsize=12)
    plt.xlabel("Type", fontsize=12)
