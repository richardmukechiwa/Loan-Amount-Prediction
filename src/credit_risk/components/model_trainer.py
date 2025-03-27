from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import joblib
import os
from sklearn.model_selection import GridSearchCV
from joblib import parallel_backend


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_data.dropna(inplace=True)
        test_data.dropna(inplace=True)

        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[self.config.target_column]
        test_y = test_data[self.config.target_column]

        numerical_features = ["Income", "Emp_length", "Rate", "Percent_income"]
        categorical_features = ["Home", "Intent"]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_features),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
            ]
        )

        # Check One-Hot Encoding Output Before Model Fitting
        train_x_transformed = preprocessor.fit_transform(train_x)
        test_x_transformed = preprocessor.transform(test_x)

        print("Transformed train_x shape:", train_x_transformed.shape)
        print("Transformed test_x shape:", test_x_transformed.shape)
        
        # Proceed with the pipeline
        model_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_leaf=self.config.min_samples_leaf,
                min_samples_split=self.config.min_samples_split,
                random_state=42))
        ])

        model_pipeline.fit(train_x, train_y)

        # Save the pipeline
        #joblib.dump(model_pipeline, os.path.join(self.config.root_dir, self.config.model_name))
        
         #model evaluation with GridSearchCV
        param_grid = {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [10, 20, 30],
            'model__min_samples_split': [2, 5, 10],
         'model__min_samples_leaf': [1, 2, 4]
        }

        
        
        grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs= -1, verbose=2)
        
        with parallel_backend('threading'):
            grid_search.fit(train_x, train_y)
          
        
        print("Best Parameters:", grid_search.best_params_) 
        
        #Best Parameters: {'model__max_depth': 30, 'model__min_samples_leaf': 1, 'model__min_samples_split': 2, 'model__n_estimators': 300}
        
        #Save the pipeline
        joblib.dump(grid_search, os.path.join(self.config.root_dir, self.config.model_name))
        
        #train_x_transformed = preprocessor.transform(train_x)  # Ensure this matches model training


        #