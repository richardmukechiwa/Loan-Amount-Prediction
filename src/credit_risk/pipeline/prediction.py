from pathlib import Path
import joblib
import pandas as pd
import os
import gdown

class PredictionPipeline:
    def __init__(self):
        # Define model path
        self.model_path = Path("artifacts/model_trainer/model.joblib")
        self.file_id = "1QAGYRh8euKBonvOrSdzPlAx_RsQDQ-jL"  # Your Google Drive file ID

        # Ensure the model is available
        self.download_model_if_needed()

        # Load the entire pipeline (preprocessor + model)
        self.pipeline = joblib.load(self.model_path)

    def download_model_if_needed(self):
        """Downloads the model from Google Drive if it's missing."""
        if not self.model_path.exists():
            print("🔽 Model not found! Downloading from Google Drive...")
            self.model_path.parent.mkdir(parents=True, exist_ok=True)  # Create directories if they don't exist
            gdown.download(f"https://drive.google.com/uc?id={self.file_id}", str(self.model_path), quiet=False)
            print("✅ Model downloaded successfully!")

    def predict(self, raw_data):
        """
        Accepts raw input data (dictionary or DataFrame),
        applies transformations, and makes predictions.
        """
        # Ensure input is a DataFrame
        if isinstance(raw_data, dict):  # If single sample, convert to DataFrame
            raw_data = pd.DataFrame([raw_data])
        elif isinstance(raw_data, list):  # If multiple samples, convert to DataFrame
            raw_data = pd.DataFrame(raw_data)
        elif not isinstance(raw_data, pd.DataFrame):  # Invalid input
            raise ValueError("Input data must be a dictionary, list, or Pandas DataFrame")

        # Apply preprocessing and make predictions
        predictions = self.pipeline.predict(raw_data)

        return predictions.tolist()  # Convert NumPy array to list for JSON response



