#!/usr/bin/env bash

set -e

echo "Downloading dataset bank-customer-churn-dataset from Kaggle"

if [ -f "data/gauravtopre/bank-customer-churn-dataset.zip" ]; then
  echo "Dataset already exists"
  echo "Skipping download..."
else
  echo "Downloading dataset..."
  kaggle datasets download -d gauravtopre/bank-customer-churn-dataset -p data/
fi

if [ -d "data/bank-customer-churn-dataset" ]; then
  echo "Dataset already unzipped"
  echo "Skipping unzip..."
else
  echo "Unzipping dataset..."
  unzip data/bank-customer-churn-dataset.zip -d data/
fi

echo "Done."
