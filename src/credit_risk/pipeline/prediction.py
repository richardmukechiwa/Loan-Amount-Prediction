from pathlib import Path
import joblib
import pandas as pd

class PredictionPipeline:
    def __init__(self):
        # Load the entire pipeline (including preprocessor + model)
        self.pipeline = joblib.load(Path("artifacts/model_trainer/model.joblib"))

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


