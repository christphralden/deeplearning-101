#!/usr/bin/env bash

set -e

echo "Downloading dataset handwritten-english-characters-and-digits from Kaggle"

if [ -f "data/handwritten-english-characters-and-digits.zip" ]; then
  echo "Dataset already exists in data/handwritten-english-characters-and-digits.zip"
  echo "Skipping download..."
else
  echo "Downloading dataset..."
  kaggle datasets download -d sujaymann/handwritten-english-characters-and-digits -p data/
fi

if [ -d "data/handwritten-english-characters-and-digits" ]; then
  echo "Dataset already unzipped in data/handwritten-english-characters-and-digits/"
  echo "Skipping unzip..."
else
  echo "Unzipping dataset..."
  unzip data/handwritten-english-characters-and-digits.zip -d data/
fi

echo "Done."
