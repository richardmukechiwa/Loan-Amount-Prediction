from pathlib import Path
import joblib
import pandas as pd

class PredictionPipeline:
    def __init__(self):
        # Load the trained pipeline (preprocessor + model)
        self.model = joblib.load(Path("artifacts/model_trainer/model.joblib"))  

    def predict(self, raw_data):
        """
        Takes raw data as a Pandas DataFrame, applies the saved pipeline, and makes predictions.
        """
        if not isinstance(raw_data, pd.DataFrame):
            raise ValueError("Input data must be a Pandas DataFrame.")
        
        # The pipeline (preprocessor + model) is already fitted, so we just use `.predict()`
        predictions = self.model.predict(raw_data)

        return predictions

