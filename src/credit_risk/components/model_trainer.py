import os
from credit_risk import logger
from sklearn.linear_model import ElasticNet 
import joblib
import pandas as pd
from credit_risk.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config=config
        
    def train(self):
        train_data=pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        
        test_data.dropna(inplace=True)
        train_data.dropna(inplace=True)
        
        print(test_data.isnull().sum())
        print(train_data.isnull().sum())
        
        
        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop(self.config.target_column, axis=1)
        train_y=train_data[[self.config.target_column]]
        test_y=test_data[[self.config.target_column]]
        
        elnet   = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42)
        elnet.fit(train_x, train_y)
    
        joblib.dump(elnet,os.path.join(self.config.root_dir, self.config.model_name))