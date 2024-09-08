from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


from tensorflow import keras
from keras.layers import Input, Dense, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy 
from keras.metrics import BinaryAccuracy, FalseNegatives, FalsePositives
from keras import Sequential

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
    debug: bool=False
) -> Tuple[pd.DataFrame, pd.Series]:
    """Loads the customer churn prediction dataset and preprocesses it.
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
        print("parse_data:") 
        print(f"Dataset Shape: {data.shape}\n")
        print(f"Columns in the dataset: {data.columns.tolist()}\n")
        print(f"Basic statistics of features: {features.describe()}\n")
        print(f"Features Shape: {features.shape}\n")
        print(f"Label Shape: {label.shape}\n")

    return features, label  

def normalize_feature_data(
    features: pd.DataFrame,
    scaler,
    debug: bool=False
) -> np.ndarray:
    """Numerical features are transformed to values on a similar scale,
    ensuring fair comparison and better performance for algorithms sensitive to the range of input data.

    Args:
        features: DataFrame which consists of numerical features
        scaler: Scaler of choice
        debug: Shows information about features

    Returns:
        features: normalized value of ndarray array of shape (n_samples, n_features_new)
    """

    normalized = scaler.fit_transform(features)

    if debug:
        print(f"normalize_feature_data:")
        feature_names = scaler.get_feature_names_out(
                input_features=features.columns
            )
        print(f"Feature names: {feature_names}\n")
        
    return normalized

def train_test_val_split(
    data: Tuple[np.ndarray, np.ndarray],
    train: float = 0.7,
    test: float = 0.2,
    val: float = 0.1,
    random_state: int = 42,
    debug: bool = False
):
    """Splits data into training, validation, and testing sets.

    Args:
        data: Tuple of features (X) and target (y) as ndarrays.
        train: Proportion of data for training.
        test: Proportion of data for testing.
        val: Proportion of data for validation.
        random_state: Random state for reproducibility.
        debug: Prints the shape of the datasets.

    Returns:
        Tuple of (x_train, x_val, x_test, y_train, y_val, y_test)
    """
    x, y = data

    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=(test + val), random_state=random_state
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=(test / (test + val)), random_state=random_state
    )

    if debug:
        print("train_test_val_split:") 
        print(f"x_train: {x_train.shape}")
        print(f"x_val: {x_val.shape}")
        print(f"x_test: {x_test.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"y_val: {y_val.shape}")
        print(f"y_test: {y_test.shape}")

    return x_train, x_val, x_test, y_train, y_val, y_test

def create_model(
    input_shape: Tuple[int],
    debug: bool=False
):
    """
    Input (8) -> Layer(128) -> Dropout(50%) -> Layer(64) -> Dropout(50%) -> Layer(32) -> Layer(1)
    """
    model = Sequential([
        Input(
            shape=input_shape # Adjust this based on feature, since we have 8 columns so the size will be 8
        ),
        Dense(
            128,
            activation='relu',
            kernel_regularizer=l2(0.00),
        ),
        Dropout(0.5),
        Dense(
            64,
            activation='relu',
            kernel_regularizer=l2(0.05),
        ),
        Dropout(0.5),
        Dense(
            32,
            activation='relu',
            kernel_regularizer=l2(0.05),
        ),
        Dense(
            1, # Here we use one since the output of the label is binary (1,0)
            activation='sigmoid',
            kernel_regularizer=l2(0.05),
        ),
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=BinaryCrossentropy(),
        metrics=[
            BinaryAccuracy(),
            FalseNegatives(),
            FalsePositives()
        ]
    )

    if debug:
            print(model.summary())

    return model

def evaluate_model(y_true, y_pred):
    """
    Prints precision, recall, and F1-Score along with the confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        None
    """
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))


def plot(history):
    plt.figure(figsize=(10, 6))

    # First subplot: Training and Validation Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['binary_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_binary_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Second subplot: Training and Validation Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()

def main():
    features, label = parse_data(debug=True)
    features = normalize_feature_data(
        features=features,
        scaler=StandardScaler(),
        debug=True
    )

    x_train, x_val, x_test, y_train, y_val, y_test = train_test_val_split(
        data=(features, label),
        debug=True
    )
    
    multi_layer_perceptron = create_model(
        input_shape=(x_train.shape[1],),
        debug=True
    )

    history = multi_layer_perceptron.fit(
        x_train,
        y_train,
        epochs=100,
        validation_data=(x_val, y_val)
    )

    # Making predictions on the test set
    predictions = multi_layer_perceptron.predict(x_test)
    predictions_labels = [1 if p >= 0.5 else 0 for p in predictions]
    
    # Printing metrics like precision, recall, and F1-score
    evaluate_model(y_test, predictions_labels)

    # Plot training and validation metrics
    plot(history)

    # Printing the testing accuracy and loss
    loss, accuracy, false_negatives, false_positives = multi_layer_perceptron.evaluate(x_test, y_test)
    print(f'Accuracy on testing: {accuracy}')
    print(f'Loss on testing: {loss}')
    print(f'False Negatives: {false_negatives}')
    print(f'False Positives: {false_positives}')

if __name__ == "__main__":
    main()

