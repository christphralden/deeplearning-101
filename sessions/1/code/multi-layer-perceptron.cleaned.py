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

def parse_data() -> Tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv('./data/Bank Customer Churn Prediction.csv')
    data = data.drop(columns=['customer_id', 'country', 'gender'])
    
    features = data.drop(columns=['churn'])

    label = data['churn']    

    return features, label  

def normalize_feature_data(
    features: pd.DataFrame,
    scaler,
) -> np.ndarray:
    return scaler.fit_transform(features)

def train_test_val_split(
    data: Tuple[np.ndarray, np.ndarray],
    train: float = 0.7,
    test: float = 0.2,
    val: float = 0.1,
    random_state: int = 42,
):
    x, y = data

    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=(test + val), random_state=random_state
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=(test / (test + val)), random_state=random_state
    )
    
    return x_train, x_val, x_test, y_train, y_val, y_test

def create_model(
    input_shape: Tuple[int],
):
    model = Sequential([
        Input(
            shape=input_shape
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
            1,  
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
        ],
    )
    
    return model

def evaluate_model(y_true, y_pred):
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))


def plot(history):
    plt.figure(figsize=(10, 6))

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
    features, label = parse_data()
    features = normalize_feature_data(
        features=features,
        scaler=StandardScaler(),
    )

    x_train, x_val, x_test, y_train, y_val, y_test = train_test_val_split(
        data=(features, label),
    )
    
    multi_layer_perceptron = create_model(
        input_shape=(x_train.shape[1],),
    )

    history = multi_layer_perceptron.fit(
        x_train,
        y_train,
        epochs=100,
        validation_data=(x_val, y_val)
    )

    predictions = multi_layer_perceptron.predict(x_test)
    predictions_labels = [1 if p >= 0.5 else 0 for p in predictions]
    
    evaluate_model(y_test, predictions_labels)

    plot(history)

    # Printing the testing accuracy and loss
    loss, accuracy, false_negatives, false_positives = multi_layer_perceptron.evaluate(x_test, y_test)
    print(f'Accuracy on testing: {accuracy}')
    print(f'Loss on testing: {loss}')
    print(f'False Negatives: {false_negatives}')
    print(f'False Positives: {false_positives}')

if __name__ == "__main__":
    main()

