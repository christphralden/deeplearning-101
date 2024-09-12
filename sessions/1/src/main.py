import tensorflow as tf

from model.mlp import MLP

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


def parse_data():
    data = pd.read_csv("./data/Bank Customer Churn Prediction.csv")
    data = data.drop(columns=["customer_id", "country", "gender"])

    label = data["churn"]
    features = data.drop(columns=["churn"])

    return features, label


def normalize_feature_data(
    features,
    scaler,
):
    return scaler.fit_transform(features)


def train_test_val_split(
    data,
    test = 0.2,
    val = 0.2,
    random_state = 42,
):
    x, y = data

    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=(test + val), random_state=random_state
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=(test / (test + val)), random_state=random_state
    )

    return x_train, x_val, x_test, y_train, y_val, y_test

def evaluate_model(y_true, y_pred):
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Negative", "Positive"]))


def plot(history):
    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["binary_accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_binary_accuracy"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

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

    mlp = MLP(output_units=1)

    mlp.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.FalseNegatives(),
            tf.keras.metrics.FalsePositives()
        ],
    )
    

    history = mlp.fit(
        x_train, y_train, epochs=100, validation_data=(x_val, y_val)
    )

    predictions = mlp.predict(x_test)
    predictions_labels = [1 if p >= 0.5 else 0 for p in predictions]

    evaluate_model(y_test, predictions_labels)

    plot(history)


if __name__ == "__main__":
    main()
