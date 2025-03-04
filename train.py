import dagshub
import mlflow

dagshub.init(repo_owner='richardmukechiwa', repo_name='Loan-Amount-Prediction', mlflow=True)

with mlflow.start_run():
    mlflow.log_param('model_type', 'Linear Regression')
    mlflow.log_metric('rmse', 0.25)

print("MLflow tracking initialized successfully!")
