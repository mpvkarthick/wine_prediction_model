# Wine Prediction Model

A machine learning project for predicting wine quality using Random Forest regression with MLflow experiment tracking and DVC data versioning.

## Overview

This project demonstrates an end-to-end ML pipeline that:
- Loads wine quality data from CSV
- Trains a Random Forest model for quality prediction
- Tracks experiments, parameters, and metrics with MLflow
- Versions datasets using DVC (Data Version Control)
- Logs detailed metadata for reproducibility and auditability

## Project Structure

```
.
├── train.py                 # Main training script
├── utils.py                 # Utility functions for data loading
├── requirements.txt         # Python dependencies
├── data/
│   ├── wine_sample.csv      # Sample wine dataset
│   └── wine_sample.csv.dvc  # DVC metadata file
├── mlfow-connect/           # MLflow connection configuration
└── README.md               # This file
```

## Requirements

- Python 3.8+
- pandas
- scikit-learn
- numpy
- joblib
- mlflow

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Wine-Prediction-Model
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Initialize DVC (optional, for data versioning):
```bash
dvc init
dvc pull  # to fetch versioned datasets
```

## Usage

### Basic Training

Run the training script with default parameters:
```bash
python train.py
```

### Custom Parameters

Train with custom hyperparameters and dataset:
```bash
python train.py \
  --csv data/wine_sample.csv \
  --target quality \
  --experiment "wine-experiments" \
  --run "rf-baseline" \
  --n-estimators 100 \
  --max-depth 10 \
  --test-size 0.3 \
  --random-state 42
```

### Available Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv` | `data/wine_sample.csv` | Path to CSV dataset |
| `--target` | `quality` | Target column name |
| `--experiment` | `first-helloworld-experiment` | MLflow experiment name |
| `--run` | `run-2` | MLflow run name |
| `--n-estimators` | `50` | Random Forest n_estimators |
| `--max-depth` | `5` | Random Forest max_depth |
| `--test-size` | `0.4` | Test split fraction (0-1) |
| `--random-state` | `42` | Random seed for reproducibility |

## MLflow Integration

### Setup

Set the MLflow tracking URI before running:
```bash
export MLFLOW_TRACKING_URI="http://127.0.0.1:8080"
python train.py
```

If not set, defaults to `http://127.0.0.1:8080`.

### Start MLflow UI

```bash
mlflow ui --host 127.0.0.1 --port 8080
```

Then visit `http://127.0.0.1:8080` to view experiments and runs.

### What Gets Tracked

**Dataset Metadata:**
- Dataset path (absolute)
- Dataset SHA256 hash (for integrity verification)
- Dataset dimensions (rows, columns)
- Target column name

**Model Parameters:**
- n_estimators
- max_depth
- test_size
- random_state
- Train/test split sizes

**Performance Metrics:**
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score

## Data Versioning with DVC

### Track a Dataset

```bash
dvc add data/wine_sample.csv
git add data/wine_sample.csv.dvc data/.gitignore
git commit -m "Add wine dataset version"
```

### Retrieve a Versioned Dataset

```bash
dvc pull
```

### Switch Between Versions

```bash
git checkout <commit-hash> data/wine_sample.csv.dvc
dvc checkout
```

## Reproducibility

The project ensures reproducibility through:

1. **Random seed control** - Use `--random-state` flag
2. **Dataset hashing** - SHA256 checksums logged with each run
3. **Parameter logging** - All hyperparameters tracked in MLflow
4. **Data versioning** - DVC tracks exact dataset versions

To reproduce a previous run:
```bash
python train.py \
  --csv data/wine_sample.csv \
  --experiment "wine-experiments" \
  --run "rf-baseline" \
  --n-estimators 100 \
  --max-depth 10 \
  --test-size 0.3 \
  --random-state 42
```

## Utilities

### `utils.py`

Helper functions for data handling:

- `load_data(path)` - Load CSV and validate 'quality' column exists
- `features_and_target(df)` - Extract features (X) and target (y) from dataframe

Example:
```python
from utils import load_data, features_and_target

df = load_data('data/wine_sample.csv')
X, y = features_and_target(df)
```

## Model Performance

The Random Forest model trains on the wine dataset to predict quality scores. Key metrics:

- **MSE**: Mean Squared Error - measures average prediction error
- **RMSE**: Root Mean Squared Error - interpretable in same units as target
- **R²**: Coefficient of Determination - proportion of variance explained (0-1)

Check MLflow UI for detailed metrics across different runs and experiments.

## Troubleshooting

### CSV not found error
Ensure `data/wine_sample.csv` exists or provide the correct path with `--csv` flag.

### Target column not found
Verify the dataset contains the target column (default: 'quality'). Use `--target` flag to specify a different column.

### MLflow connection error
- Ensure MLflow tracking server is running: `mlflow ui`
- Check `MLFLOW_TRACKING_URI` environment variable is set correctly
- Default URI: `http://127.0.0.1:8080`

### DVC errors
- Run `dvc init` if `.dvc/` directory doesn't exist
- Check remote storage configuration: `dvc remote list`

## Contributing

1. Create a feature branch
2. Make changes and test with `python train.py`
3. Verify MLflow logging works correctly
4. Commit with clear messages
5. Submit pull request

## License

This project is licensed under the MIT License

## Contact

For questions or issues, please reach out to the project maintainer.
