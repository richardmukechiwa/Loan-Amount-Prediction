# Loan-Amount-Prediction-Project

## Workflows

1. Update config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update main.py
9. Update the app.py

# How to run?

### STEPS:

Clone the repository

```python
https://github.com/richardmukechiwa/Loan-Amount-Prediction
```

### STEP 01- Create a conda environment after opening the repository

```python
conda create -n ln python=3.10 -y
```

```python
conda activate ln
```

### STEP 02- install the requirements

```python
pip install -r requirements.txt
```

```python
#Finally run the following command
python app.py
```

Then,
```python
open up your local host and port
```

### MLflow

[Documentation](https://mlflow.org/docs/latest/index.html)

##### cmd
- mlflow ui


### dagshub

[dagshub](https://dagshub.com/)

https://dagshub.com/richardmukechiwa/Loan-Amount-Prediction.mlflow

import dagshub
dagshub.init(repo_owner='richardmukechiwa', repo_name='Loan-Amount-Prediction', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)

Run this to export as env variables:

```python

import dagshub
import mlflow

# Initialize DagsHub MLflow tracking (automatically sets tracking URI)
dagshub.init(repo_owner='richardmukechiwa', repo_name='Loan-Amount-Prediction', mlflow=True)

# Start an MLflow experiment
with mlflow.start_run():
    mlflow.log_param('model_type', 'Linear Regression')
    mlflow.log_metric('rmse', 0.25)  # Example metric

print("MLflow tracking initialized successfully!")
```

