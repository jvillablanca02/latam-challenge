import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

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
    plt.show()

    # Distribution by Destination
    flight_by_destination = data['SIGLADES'].value_counts()
    plt.figure(figsize=(10, 2))
    sns.set(style="darkgrid")
    sns.barplot(x=flight_by_destination.index, y=flight_by_destination.values, color="lightblue", alpha=0.8)
    plt.title("Flight by Destination")
    plt.ylabel("Flights", fontsize=12)
    plt.xlabel("Destination", fontsize=12)
    plt.xticks(rotation=90)
    plt.show()
