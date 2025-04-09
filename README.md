
## Loan Amount Prediction - End-to-End Machine Learning Project

## Workflows

1. Update config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update main.py
9. Update the app.py

### Problem Statement
In the financial sector, accurately predicting the amount a client can borrow is critical for mitigating risks and ensuring responsible lending.This project focuses on predicting the loan amount an applicant is likely to receive based on features such as income, credit history, loan term, and more. It demonstrates my ability to handle real-world financial data, perform data preprocessing, build regression models, and deploy the final solution as an interactive web app using Streamlit.

### Tech Stack & Tools
- **Language**: Python 3
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost, imbalanced-learn
- **Pipeline**: Modular structure with clear separation for data ingestion, transformation, training, evaluation, and prediction
- **Logging**: Custom logging set up for traceability
- **Model Tracking**: Pickle for model persistence
- **App Interface**: Streamlit

### Folder Structure
```
Loan-Amount-Prediction/
│
├── artifacts/                # Stores processed data
├── credit_risk/             # Core package (data ingestion, transformation, model code)
│   ├── components/
│   ├── config/
│   ├── pipeline/
│   └── utils.py
├── notebook/                # Jupyter notebooks for exploration and trials
├── saved_models/            # Trained model pickle files
├── static/                  # Visual assets for Streamlit app
├── templates/               # HTML templates
├── Dockerfile
├── app.py                   # Streamlit application
├── main.py                  # Entry point for training pipeline
├── requirements.txt
└── setup.py                 # For packaging
```

### Key Highlights 

- __Situation:__ Financial institutions need to assess loan eligibility and amount using applicant data.

- __Task:__ Build a regression model to predict the loan amount using features from a publicly available dataset.

- __Action:__ Performed EDA, cleaned data, encoded categorical variables, and built a Random Forest Regression model. Deployed the app using Streamlit and hosted it on Streamlit Cloud.

- __Result:__ Achieved an R² score of 0.99 on test data, indicating a strong fit. The model can assist loan officers in making informed decisions.

###  How to Run
1. Clone the repository
```bash
git clone https://github.com/richardmukechiwa/Loan-Amount-Prediction.git
```
2. Navigate to the directory and install dependencies:
```bash
cd Loan-Amount-Prediction
pip install -r requirements.txt
```
3. Train the model:
```bash
python main.py
```
4. Launch the Streamlit app:
```bash
streamlit run app.py
```

### Features of the Streamlit App

- **Loan Amount Prediction**: Users can input their financial details to estimate the loan amount they may qualify for.
- **Model Training**: Retrain the machine learning model with updated data directly from the app.
- **Documentation**: View detailed information about the app and its functionality.
- **Source Code Access**: Direct link to the GitHub repository for developers.
- **Contact Information**: Reach out to the developer for inquiries or feedback.

##  App Demo
Here is a preview of the app:

[Loan Prediction Demo](https://github.com/richardmukechiwa/Datasets/raw/refs/heads/main/Loan-App.mp4)

## Example Input and Output

Input

Income: $50,000

Employment Length: 5 years

Interest Rate: 10%

Percent of Income for Loan: 20%

Credit Length: 10 years

Home Ownership: MORTGAGE

Loan Intent: PERSONAL

Output

Predicted Loan Amount: $30,000



### Track experiments using MLflow:
   - Configure MLflow to log experiments to DagsHub:

import mlflow
mlflow.set_tracking_uri("https://dagshub.com/richardmukechiwa/Loan-Amount-Prediction.mlflow") 


### Docker Support
Build the Docker image and run the app in a container:
```bash
docker build -t loan_app .
docker run -p 8501:8501 loan_app
```

### Model Performance
- ML Model: Random Forest Regressor
- R² Score: 0.99 on test data
- RMSE: ~3000 USD

###  Acknowledgements
Special thanks to mentors Krish Naik, Bappy Hamed, and Kasim Ali for their invaluable guidance throughout this project.

###  Connect
For collaboration or questions, reach out on [LinkedIn](https://www.linkedin.com/in/richardmukechiwa/)


### License
This project is licensed under the MIT License. See the LICENSE file for details.

