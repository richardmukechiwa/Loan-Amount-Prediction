import joblib
import numpy as np
from pathlib import Path

class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path("artifacts/model_trainer/model.joblib"))  # Load trained model
        
        # Load the saved ColumnTransformer
        self.preprocessor = joblib.load(Path("artifacts/model_trainer/preprocessor.joblib"))

    def predict(self, raw_data):
        # Apply the saved preprocessor to incoming raw data
        transformed_data = self.preprocessor.transform(raw_data)
        
        # Predict on transformed data
        scaled_prediction = self.model.predict(transformed_data)
        return scaled_prediction

