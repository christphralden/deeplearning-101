# That Time I Created A Neural Network From Scratch (TTICANNFS)

```bash
# Unix
python -m venv venv
source venv/bin/activate


pip install -r requirements.txt

# Allow the script to exectue
chmod +x ./kaggle.sh

# Execute script to download dataset
./kaggle.sh
```

# Commands

To run you can use CLI commands

---

### dataset

Only to load the dataset, does nothing else. Probably if you want to view the dataset.

_Usage_

```bash
manage.py dataset [-h] [--show] [--with-image] [--alphanumeric]
```

_Options:_

- `-h, --help`: show this help message and exit
- `--show`: Show dataset statistics
- `--with-image`: Display images from the dataset
- `--alphanumeric`: Adds the alphanumeric dataset

---
