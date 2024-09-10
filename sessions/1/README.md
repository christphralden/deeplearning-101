## train_test_val_split

- x_train: (6000, 8)
- x_val: (2000, 8)
- x_test: (2000, 8)
- y_train: (6000,)
- y_val: (2000,)
- y_test: (2000,)

## Model:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dense (Dense)                        │ (None, 256)                 │           2,304 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization                  │ (None, 256)                 │           1,024 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 256)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 128)                 │          32,896 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_1                │ (None, 128)                 │             512 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (Dropout)                  │ (None, 128)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 64)                  │           8,256 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_2                │ (None, 64)                  │             256 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_2 (Dropout)                  │ (None, 64)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_3 (Dense)                      │ (None, 32)                  │           2,080 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_3                │ (None, 32)                  │             128 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_3 (Dropout)                  │ (None, 32)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_4 (Dense)                      │ (None, 1)                   │              33 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
```

- Total params: `47,489 (185.50 KB)`
- Trainable params: `46,529 (181.75 KB)`
- Non-trainable params: `960 (3.75 KB)`

## Results

## Sequential Model

```
Classification Report:
              precision    recall  f1-score   support

    Negative       0.87      0.96      0.91      1570
    Positive       0.77      0.46      0.58       430

    accuracy                           0.85      2000
   macro avg       0.82      0.71      0.74      2000
weighted avg       0.85      0.85      0.84      2000

```

<p align="center">
  <img src="./_attachments/sequential_model.png" alt="sequential_model" />
</p>

<div align="center">
  <em>Sequential Model Training, Validation Accuracy and Loss</em>
</div>

## Sub Class Model

```
Classification Report:
              precision    recall  f1-score   support

    Negative       0.86      0.97      0.91      1570
    Positive       0.78      0.44      0.56       430

    accuracy                           0.85      2000
   macro avg       0.82      0.70      0.74      2000
weighted avg       0.85      0.85      0.84      2000

```

<p align="center">
  <img src="./_attachments/model.png" alt="sub_class_model" />
</p>

<div align="center">
  <em>Sub Class Model Training, Validation Accuracy and Loss</em>
</div>
