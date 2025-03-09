# ML flow Tracking
import mlflow
import dagshub

# Initialize DagsHub MLflow tracking
dagshub.init(repo_owner="richardmukechiwa", repo_name="Loan-Amount-Prediction", mlflow=True)

with mlflow.start_run():
    mlflow.log_param("batch_size", 32)
    mlflow.log_metric("loss", 0.23)
    
# To run the experiment, use the following command in the terminal:
# python run_experiment.py
