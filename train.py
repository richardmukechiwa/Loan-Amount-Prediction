# Mlflow tracking URI
import mlflow
import dagshub

# Initialize DagsHub MLflow tracking
dagshub.init(repo_owner="richardmukechiwa", repo_name="Loan-Amount-Prediction", mlflow=True)

with mlflow.start_run():
    mlflow.log_param("alpha", 0.8)
    mlflow.log_param("l1_ratio", 0.8)
    mlflow.log_metric("rmse", 0.92)

#To run the mlflow experiment
# run the following command in your terminal
# python train.py