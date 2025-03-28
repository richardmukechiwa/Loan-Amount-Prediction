import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import logging

logger = logging.getLogger(__name__)

class DataTransformationConfig:
    def __init__(self, data_path, root_dir):
        self.data_path = data_path
        #self.model_path = model_path
        self.root_dir = root_dir
        

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
    
    def data_cleaning(self):
        data = pd.read_csv(self.config.data_path)
        
        # Remove columns which are not necessary for the analysis
        data.drop(columns=["Id", "Status", "Default"], inplace=True)
        
        # Drop null values
        data.dropna(inplace=True)
        
        logger.info("Null values dropped")
        
        # Remove outliers
        data = data[(data['Age'] < 60) & (data['Emp_length'] <= 10) & (data['Income'] < 200000)]
        
        logger.info("Data cleaning complete")
        
        return data
    
    def exploratory_data_analysis(self, data):
        # Check descriptive statistics
        print(data.describe())
        
        # Check non-numeric columns
        print(data.describe(include='object'))
        
        # Check the target variable
        data['Amount'].hist()
        plt.ylabel('Count')
        plt.xlabel('Amount')    
        plt.title('Loan Amount Distribution')
        plt.show()
        
        print("The distribution is right-skewed, meaning most loan amounts fall in the lower range (below 10,000), while fewer loans exist at higher amounts.")
        
        # Calculate Amount distribution by Age   
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x='Age', y='Amount', data=data) 
        plt.xlabel('Age')
        plt.ylabel('Amount')            
        plt.title('Loan Amount by Age')
        plt.show()
        
        # Calculate Amount distribution by Income
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x='Income', y='Amount', data=data)    
        plt.xlabel('Income')
        plt.ylabel('Amount')
        plt.title('Loan Amount by Income')
        plt.show()
        
        # Loan purpose count
        plt.figure(figsize=(12, 6))
        data["Intent"].value_counts().plot(kind='bar')
        plt.ylabel('Count')
        plt.xlabel('Intent')
        plt.title('Loan Intent Distribution')
        plt.show()
        
        # Check multicollinearity and correlation
        plt.figure(figsize=(12, 6))  
        corr = data.select_dtypes(include=['int64', 'float64']).drop('Amount', axis=1).corr()    
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()
        
        #drop 'Age'
        data.drop('Age', axis=1, inplace=True)
        
         # dropping columns after feature selection, #columns to be dropped: Home, Rate, Percent_income, Cred_length
        #data.drop(['Home', 'Rate', 'Percent_income', 'Cred_length'], axis=1, inplace=True)
        
        
        return data
    
    def train_test_splitting(self, data):
        
        
        # Split the data into train and test
        train, test = train_test_split(data, test_size=0.2, random_state=42)  
        
        train.to_csv(os.path.join(self.config.root_dir, 'train.csv'), index=False)
        test.to_csv(os.path.join(self.config.root_dir, 'test.csv'), index=False) 
        
        # Save the train and test data to the root directory
        logger.info("Data split into train and test data")  
        logger.info(f"Train data shape: {train.shape}")         
        logger.info(f"Test data shape: {test.shape}")  
        
        print(train.shape)
        print(test.shape)
        
        
        return train, test

    

    

