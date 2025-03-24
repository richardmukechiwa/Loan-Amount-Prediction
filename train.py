# Mlflow tracking URI
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
# run the following command in your terminal
# python train.py