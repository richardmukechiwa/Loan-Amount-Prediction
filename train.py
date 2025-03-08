import dagshub
import mlflow
import os

# Set up connection to DagsHub MLflow
dagshub.init(repo_owner="richardmukechiwa", repo_name="Loan-Amount-Prediction", mlflow=True)

# Set MLflow tracking URI
mlflow.set_tracking_uri("https://dagshub.com/richardmukechiwa/Loan-Amount-Prediction.mlflow")

# Authenticate using environment variables
os.environ["MLFLOW_TRACKING_USERNAME"] = "richardmukechiwa"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "eb0d37817b53655929466e335a4b93d0a38eb874"


import dagshub
dagshub.init(repo_owner='richardmukechiwa', repo_name='Loan-Amount-Prediction', mlflow=True)

import mlflow
with mlflow.start_run():
  #mlflow.log_param('', 'value')
  mlflow.log_metric('r2', 0.75)
