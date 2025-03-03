import os
from credit_risk import logger
from sklearn.linear_model import LinearRegression 
import joblib
import pandas as pd
from credit_risk.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config=config
        
    def train(self):
        train_data=pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        
        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop(self.config.target_column, axis=1)
        train_y=train_data[[self.config.target_column]]
        test_y=test_data[[self.config.target_column]]
        
        reg = LinearRegression(fit_intercept=self.config.fit_intercept, n_jobs=self.config.n_jobs)
        reg.fit(train_x, train_y)
    
        joblib.dump(reg, os.path.join(self.config.root_dir, self.config.model_name))