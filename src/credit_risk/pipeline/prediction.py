from pathlib import Path
import joblib
import numpy as np


class PredictionPipeline:
    def __init__(self):
        self.pipeline = joblib.load(Path("artifacts/model_trainer/pipeline.joblib"))
        self.model = joblib.load(Path("artifacts/model_trainer/model.joblib"))  # Load trained model

    def predict(self, raw_data):
        # Apply the saved pipeline (which includes the encoder and scaler) on the raw data
        data = self.pipeline.transform(raw_data)

        # Make prediction using the loaded model and return predictions
        predictions = self.model.predict(data)

        # Return predictions
        return predictions
