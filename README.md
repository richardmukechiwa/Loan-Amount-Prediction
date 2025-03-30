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

# Loan Amount Prediction App

![Loan Amount Prediction](https://loan-amount-prediction-v5wxdubjubukkct6memh5n.streamlit.app/)

## 📖 Overview
The **Loan Amount Prediction App** is a machine learning-powered web application that predicts the loan amount a borrower may qualify for based on their financial profile. The app is built using **Streamlit** for the frontend and integrates a trained machine learning pipeline for predictions.


## 🚀 Features
- **Loan Amount Prediction**: Users can input their financial details to estimate the loan amount they may qualify for.
- **Model Training**: Retrain the machine learning model with updated data directly from the app.
- **Documentation**: View detailed information about the app and its functionality.
- **Source Code Access**: Direct link to the GitHub repository for developers.
- **Contact Information**: Reach out to the developer for inquiries or feedback.

---

## 🛠️ Technologies Used
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**: Python
- **Machine Learning**: Scikit-learn
- **Model Serialization**: Joblib
- **Data Handling**: Pandas, NumPy
- **Deployment**: Docker (optional)

---

## 📂 Project Structure

Loan-Amount-Prediction/ ├── artifacts/ # Contains the trained model and other artifacts │ └── model_trainer/ │ └── model.joblib # Serialized machine learning model ├── src/ # Source code for the app │ └── credit_risk/ │ ├── components/ # ML components (e.g., data ingestion, model trainer) │ ├── pipeline/ # Prediction pipeline │ ├── utils/ # Utility functions ├── config/ # Configuration files │ ├── config.yaml # App configuration │ └── params.yaml # Model parameters ├── research/ # Jupyter notebooks for experimentation │ └── trials.ipynb ├── Dockerfile # Docker configuration for deployment ├── requirements.txt # Python dependencies ├── setup.py # Package setup file ├── pyproject.toml # Build system configuration ├── loan_app.py # Main Streamlit app └── README.md

# Project documentation


---

## 📊 How It Works
1. **Prediction**:
   - Users provide financial details such as income, employment length, interest rate, and loan intent.
   - The app preprocesses the input data and uses a trained machine learning model to predict the loan amount.

2. **Model Training**:
   - The app allows retraining of the model using updated data.
   - The training process is triggered by running `main.py` from the app.

---

## 🖥️ Running the App Locally

### Prerequisites
- Python 3.9 or higher
- Virtual environment (optional but recommended)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/richardmukechiwa/Loan-Amount-Prediction.git
   cd Loan-Amount-Prediction

2. Install dependencies:

pip install -r requirements.txt

3. Run the app:
streamlit run loan_app.py

4. Open the app in your browser at
   
   ```
   http://localhost:8501
   
   ```
   

🐳 Running with Docker

1. Build the Docker image:
   
  docker build -t loan-amount-prediction.
  
2. Run the Docker container:

   docker run -p 8501:8501 loan-amount-prediction

3. Access the app at
   ```python
   http://localhost:8501
   ```

📁 Key Files

- loan_app.py: Main Streamlit app for user interaction.

- artifacts/model_trainer/model.joblib: Trained machine learning model.
  
- src/credit_risk/pipeline/prediction.py: Prediction pipeline for processing input and 
  generating predictions.

- main.py: Script for training the machine learning model.
  
- requirements.txt: List of Python dependencies.
  
- Dockerfile: Configuration for containerizing the app.

📊 DagsHub Integration

This project uses DagsHub for:

- Dataset Versioning: Track and version datasets used for training.

- Model Versioning: Store and version trained models for reproducibility.
  
- Experiment Tracking: Log and visualize experiments, including hyperparameters and metrics.
  
  How to Access the DagsHub Repository
  
- You can access the DagsHub repository for this project here: DagsHub Repository

How to Use DagsHub in This Project

1. Clone the DagsHub repository:

dagshub clone richardm/Loan-Amount-Prediction
cd Loan-Amount-Prediction

2. Push datasets or models to DagsHub:

   git add data/
git commit -m "Add dataset"
git push origin main

3. Track experiments using MLflow:
   - Configure MLflow to log experiments to DagsHub:

import mlflow
mlflow.set_tracking_uri("https://dagshub.com/richardmukechiwa/Loan-Amount-Prediction.mlflow")  


🧪 Example Input and Output
Input
Income: $50,000
Employment Length: 5 years
Interest Rate: 10%
Percent of Income for Loan: 20%
Credit Length: 10 years
Home Ownership: MORTGAGE
Loan Intent: PERSONAL
Output
Predicted Loan Amount: $15,000

📜 Documentation
For detailed documentation, visit the Documentation section in the app.

🔗 Links
GitHub Repository: Loan Amount Prediction Repo
Contact: mukechiwarichard@gmail.com

🤝 Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

📧 Contact
For inquiries, reach out via email at mukechiwarichard@gmail.com.

📝 License
This project is licensed under the MIT License. See the LICENSE file for details.



---


