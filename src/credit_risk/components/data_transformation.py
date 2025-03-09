import os
import pandas as pd 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns 
   #transform categorical data and stardardize the data
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline   
from sklearn.preprocessing import OneHotEncoder
from credit_risk import logger
from credit_risk.entity.config_entity import DataTransformationConfig    



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
     # add EDA to the data 
    def data_cleaning(self):
        data = pd.read_csv(self.config.data_path)
        
        #remove  columns which are not necessary for the analysis
        data.drop(columns = ["Id","Status", "Default"], inplace=True)
        
        
        data.dropna(inplace=True)
        
        print("...............................................................")
        #drop null values
        print(data.isnull().sum())
        
        print("...............................................................")
        
        
        
        logger.info(f"Null values dropped")
        
        print("................................................................")
        print()
        
        
        
        #remove outliers
        data = data[(data['Age'] < 80) & (data['Emp_length'] < 10) & (data['Income'] < 948000)]
        
        
        print("data.head()")
        
        print(data.head())
        
        print("...................................................................")
        
        data1  = data
        
        return data1

        logger.info(f"Data cleaning complete")
        
    
      
        
    def exploratory_data_analysis(self):
        data1 = pd.read_csv(self.config.data_path)
        
        data1 = data1[(data1['Age'] < 80) & (data1['Emp_length'] < 10) & (data1['Income'] < 948000)]
        
        #check descriptive statistics
        
        print(data1.describe())
        
        #check non numeric columns
        print(data1.describe(include='object'))
        
        #check the target variable
        data1['Amount'].hist()
        plt.ylabel('Count')
        plt.xlabel('Amount')    
        plt.title('Loan Amount Distribution')
        
        
        print("The distribution is right-skewed, meaning most loan amounts fall in the lower range (below 10,000), while fewer loans exist at higher amounts");
                
        #calculate Amount distribution by Age   
        plt.figure(figsize=(12,6))
        sns.scatterplot(x='Age', y='Amount', data=data1) 
        plt.xlabel('Age')
        plt.ylabel('Amount')            
        plt.title('Loan Amount by Age'); 
        
        # calculating Amount distribution by Income
        plt.figure(figsize=(12,6))
        sns.scatterplot(x='Income', y='Amount', data=data1)    
        plt.xlabel('Income')
        plt.ylabel('Amount')
        plt.title('Loan Amount by Income');
        
        #loan purpose count
        plt.figure(figsize=(12,6))
        data1["Intent"].value_counts().plot(kind='bar')
        plt.ylabel('Count')
        plt.xlabel('Intent')
        plt.title('Loan Intent Distribution');
        
        #check multicollinearity and correlation
        plt.figure(figsize=(12,6))  
        corr = data1.select_dtypes(include=['int64', 'float64']).drop('Amount', axis=1).corr()    
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix');
        
        #drop column Id and Cred_length , Status, Default   
        columns_to_drop = ["Id", "Cred_length", "Status", "Default"]
        data1.drop(columns=[col for col in columns_to_drop if col in data1.columns], inplace=True)
        pd.options.mode.copy_on_write=True
        print(data1.head())
        
        data2 = data1
        
        return data2
        
    def feat_engineering(self):
        data2 = pd.read_csv(self.config.data_path)
        
        #feature engineering
        cat_features    = data2[["Home", "Intent"]]
        num_features   = data2[["Age",	"Income", "Emp_length", "Amount","Rate", 	"Percent_income"]]
        
        #instantiate SimpleImputer
       
        
        # instantiate the column StandardScaler
        numerical_processor = Pipeline(
            steps =[("standard scaling",  StandardScaler()
            )]  
        )
        
        # instantiate the column OneHotEncoder
        categorical_processor = Pipeline(
            steps =[("one hot encoding", OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            )]  
        )
    
        #implement the column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                
                ("numerical", numerical_processor, num_features.columns),
                ("categorical", categorical_processor, cat_features.columns)
            ]
        )   
 
        #fit the preprocessor
        preprocessor.fit(data2)
        
        
        #transform the data
        transformed_data=preprocessor.transform(data2)  

        data2=pd.DataFrame(transformed_data)
        data2.columns = num_features.columns.tolist() + preprocessor.named_transformers_["categorical"]["one hot encoding"].get_feature_names_out().tolist()
        data2.to_csv("artifacts/data_ingestion/credit_risk.csv", index=False)     
        
      

        data3 = data2
        
        return data3 
            
        
        
    def train_test_splitting(self):
        data3 = pd.read_csv(self.config.data_path)
        
        #split the data into train and test
        train, test = train_test_split(data3, test_size=0.2, random_state=42)  
        
        train.to_csv(os.path.join(self.config.root_dir, 'train.csv'), index=False)
        test.to_csv(os.path.join(self.config.root_dir, 'test.csv'), index=False)        #save the train and test data to the root directory     
        
        logger.info("Data split into train and test data")  
        logger.info(f"Train data shape: {train.shape}")         
        logger.info(f"Test data shape: {test.shape}")  
        
        print(train.shape)
        print(test.shape) 