from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
"""
churn:
phenomenon where customers terminate their relationship with a business or organization.


customer_id:        account number (unused)    
credit_score:       Credit Score
country:            string (unused)
gender:             string (unused)
age:                Age of account holder
tenure:             From how many years he/she is having bank acc in ABC Bank
balance:            Account Balance
products_number:    Number of Product from bank
credit_card:        Does this customer have credit card
active_member:      Is an active member of bank
estimated_salary:   Salary of account holder
churn:              Churn Status: used as the target. 1 if the client has left the bank during some period or 0 if he/she has not.
Aim is to Predict the Customer Churn for ABC Bank.
"""

def parse_data(
    debug: bool = False
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads the customer churn prediction dataset and preprocesses it.
    Drops unused columns (strings) and splits the label from the original dataset.

    Args:
        debug: Shows informations about the dataset

    Returns:
        tuple: A tuple containing:
            - features (pd.DataFrame): Features for the prediction (excluding the 'churn' label).
            - label (pd.Series): The target variable 'churn'.
    """
    data = pd.read_csv('./data/Bank Customer Churn Prediction.csv')
    data = data.drop(columns=['customer_id', 'country', 'gender'])
    
    features = data.drop(columns=['churn'])

    label = data['churn']    

    if debug:
        print(f"Dataset Shape: {data.shape}")
        print(f"\nColumns in the dataset: {data.columns.tolist()}")
        print(f"\nBasic statistics of features: {features.describe()}\n")
        print(f"\nFeatures Shape: {features.shape}")
        print(f"\nLabel Shape: {label.shape}")


    return features, label  

def normalize_feature_data(
    features: pd.DataFrame,
    scaler
):
    """
    Numerical features are transformed to values on a similar scale,
    ensuring fair comparison and better performance for algorithms sensitive to the range of input data.

    Args:
        features: DataFrame which consists of numerical features

    Returns:
        features: (pd.DataFrame) normalized value
    """
    return scaler.fit_transform(features)


def main():
    features, data = parse_data(debug=True)
    features = normalize_feature_data(
        features=features,
        scaler = StandardScaler()
    )
    print(features)

if __name__ == "__main__":
    main()

