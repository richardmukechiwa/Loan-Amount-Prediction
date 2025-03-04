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
        
        df = pd.read_csv("artifacts/data_ingestion/credit_risk.csv")
        df.head()
        
        df.shape
        
        df.info()
        
        df.isnull().sum()
        
        #drop the missing values
        df = df.dropna()
        
        #check missing values
        df.isnull().sum()
        
        #check descriptive statistics
        df.describe()
        
        #check non numeric columns
        #df.describe(include='object')
        
        #check the target variable
        df['Amount'].hist()
        plt.ylabel('Count')
        plt.xlabel('Amount')    
        plt.title('Loan Amount Distribution');
        #The distribution is right-skewed, meaning most loan amounts fall in the lower range (below 10,000), while fewer loans exist at higher amounts
        
        #calculate Amount distribution by Age   
        plt.figure(figsize=(12,6))
        sns.scatterplot(x='Age', y='Amount', data=df) 
        plt.xlabel('Age')
        plt.ylabel('Amount')            
        plt.title('Loan Amount by Age');    
        #Most of the loan applicants are between the age 23 to 45 years and there are three outliers above 100 years 
        
        # calculating Amount distribution by Income
        plt.figure(figsize=(12,6))
        sns.scatterplot(x='Income', y='Amount', data=df)    
        plt.xlabel('Income')
        plt.ylabel('Amount')
        plt.title('Loan Amount by Income');
        
        
        # most of the income values are concentrated on the left side (closer to zero).

        # A few extreme outliers have very high incomes (above $1M)

        # Loan amounts do not seem to increase proportionally with income, even those with high income are taking loans of varying amounts
                
        #loan purpose count
        plt.figure(figsize=(12,6))
        df['Intent'].value_counts().plot(kind='bar')
        plt.ylabel('Count')
        plt.xlabel('Intent')
        plt.title('Loan Intent Distribution'); 
        #Most of the loan applications are going towards education followed by medical, venture, personal, debt consolidation and the least number of applications are for homeimprovements.
        
        
        #check multi collinearity and correlation
        plt.figure(figsize=(12,6))  
        corr = df.select_dtypes(include=['int64', 'float64']).drop('Amount', axis=1).corr()    
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix');
        #There is a strong correlatedness between Age and Cred_length
        
        #drop column Id and Cred_length , Status, Default   
        columns_to_drop = ["Id", "Cred_length", "Status", "Default"]
        df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
        pd.options.mode.copy_on_write=True
        df.head()
        
        #feature engineering
        cat_features    = df[["Home", "Intent"]]
        num_features   = df[["Age",	"Income", "Emp_length", "Amount",	"Rate", 	"Percent_income"]]
        
        #transform categorical data and stardardize the data
        from sklearn.preprocessing import StandardScaler
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline   
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder
        
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
        preprocessor.fit(df)
        
        #transform the data
        transformed_data=preprocessor.transform(df)  

        data=pd.DataFrame(transformed_data)
        data.columns = num_features.columns.tolist() + preprocessor.named_transformers_["categorical"]["one hot encoding"].get_feature_names_out().tolist()
        data.to_csv("artifacts/data_ingestion/credit_risk_1.csv", index=False)     

        data.head()
       
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