from pathlib import Path
import joblib
import numpy as np


class PredictionPipeline:
    def __init__(self):
        self.ohe = joblib.load(Path("artifacts/model_trainer/onehotencoder.joblib"))
         
        self.scaler = joblib.load(Path("artifacts/model_trainer/preprocessor.joblib"))
        self.model = joblib.load(Path("artifacts/model_trainer/model.joblib"))  # Load trained model
        
    def predict(self, raw_data):
        # Apply the saved ohe and scaler on the raw data
        data = self.ohe.transform(raw_data)
    
        data = self.scaler.transform(data)
    
        #make prediction using the loaded model and return unscaled predictions
        predictions = self.model.predict(data)
    
        #return unscaled predictions
        return predictions
        
        
        
        
        
        
        
        # Predict on transformed data
        scaled_prediction = self.model.predict(transformed_data)
        return scaled_prediction

