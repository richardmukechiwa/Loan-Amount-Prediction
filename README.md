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
# Run the following command to start the application    

open up your local host and port

```

### dagshub

[dagshub](https://dagshub.com/)

https://dagshub.com/richardmukechiwa/Loan-Amount-Prediction.mlflow

### Initialize DagsHub MLflow tracking
```python
import mlflow
import dagshub

# Initialize DagsHub MLflow tracking
dagshub.init(repo_owner="richardmukechiwa", repo_name="Loan-Amount-Prediction", mlflow=True)

with mlflow.start_run():
    mlflow.log_param("n_estimators", 300)
    mlflow.log_param("max_depth", 30)
    mlflow.log_param("min_samples_split", 2)
    mlflow.log_param("min_samples_leaf", 1)
    mlflow.log_metric("r2", 0.9973757800995268)

#To run the mlflow experiment
#run the following command in your terminal
#python train.py
```
```python
### Creating docker image

# Dockerfile

FROM python:3.8

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

CMD ["python", "app.py"]
```

## Model File

The trained model can be downloaded from [Google Drive](https://drive.google.com/file/d/1QAGYRh8euKBonvOrSdzPlAx_RsQDQ-jL/view?usp=drive_link).




