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

MLFLOW_TRACKING_URL=https://dagshub.com/https://github.com/richardmukechiwa/Loan-Amount-Prediction\
MLFLOW_TRACKING_USERNAME=richardmukechiwa \
MLFLOW_TRACKING_PASSWORD=
python script.py

Run this to export as env variables:

```python

export MLFLOW_TRACKING_URL=https://dagshub.com/https://github.com/richardmukechiwa/Loan-Amount-Prediction

export MLFLOW_TRACKING_USERNAME=richard mukechiwa

export MLFLOW_TRACKING_PASSWORD = 

```

