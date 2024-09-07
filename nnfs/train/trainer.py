from data import get_dataset

x_train, x_test, y_train, y_test = get_dataset(train_size=10000)

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")
