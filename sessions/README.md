# Deep Learning 101

### 1. Activate venv (or conda) to not mess up your python deps

```bash
# Unix
python -m venv venv
source venv/bin/activate


pip install -r requirements.txt
```

### 2. `cd` into directories

```bash
# Example: Session 1

cd 1
```

Everything is local to corresponding session directories, but the venv is global for all sessions

```plaintext
[ 128]  .
├── [ 160]  ./code
│   ├── [ 160]  ./code/data
│   │   ├── [548K]  ./code/data/Bank Customer Churn Prediction.csv
│   │   ├── [187K]  ./code/data/bank-customer-churn-dataset.zip
│   │   └── [ 516]  ./code/data/description.txt
│   ├── [ 590]  ./code/kaggle.sh
│   └── [2.7K]  ./code/multi-layer-perceptron.py
└── [ 160]  ./neural-network-foundations
    ├── [4.7K]  ./neural-network-foundations/1_perceptron.md
    ├── [6.2K]  ./neural-network-foundations/2_activation-functions.md
    └── [ 160]  ./neural-network-foundations/_attachments
        ├── [602K]  ./neural-network-foundations/_attachments/activation-functions.png
        ├── [ 71K]  ./neural-network-foundations/_attachments/perceptron-architecture.png
        └── [ 29K]  ./neural-network-foundations/_attachments/perceptron.jpg
```

### 3. Datasets

If there are any datasets used it will be located in the `./data` relative to each session

**Important:**
In the root of each session there will be an associated `kaggle.sh` script to download the dataset. <isn>Run that first.</isn>

```bash
# Unix

# Make the script executable
chmod +x ./kaggle.sh

# Execute the script
kaggle.sh
```
