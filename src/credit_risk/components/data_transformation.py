import os
import pandas as pd 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns 
from credit_risk.entity.config_entity import DataTransformationConfig    
from credit_risk import logger


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
       
    def train_test_splitting(self):
        data = pd.read_csv(self.config.data_path)
        
        #split the data into train and test
        train, test = train_test_split(data, test_size=0.2, random_state=42)  
        
        train.to_csv(os.path.join(self.config.root_dir, 'train.csv'), index=False)
        test.to_csv(os.path.join(self.config.root_dir, 'test.csv'), index=False)        #save the train and test data to the root directory     
        
        logger.info("Data split into train and test data")  
        logger.info(f"Train data shape: {train.shape}")         
        logger.info(f"Test data shape: {test.shape}")  
        
        print(train.shape)
        print(test.shape)  