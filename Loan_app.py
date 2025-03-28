import os
import logging
import streamlit as st
import pandas as pd
from credit_risk.pipeline.prediction import PredictionPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model():
    try:
        os.system("python main.py")
        logger.info("Training completed successfully.")
        st.success("Training Successful!")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        st.error(f"Training failed: {str(e)}")

def predict_credit_risk():
    try:
        st.subheader("Enter Details for Prediction")
        
        Income = st.number_input("Income", min_value=0.0, step=100.0)
        Emp_length = st.number_input("Employment Length (Years)", min_value=0.0, step=1.0)
        Rate = st.number_input("Interest Rate", min_value=0.0, step=0.1)
        Percent_income = st.number_input("Percent of Income", min_value=0.0, step=0.1)
        Cred_length = st.number_input("Credit Length (Years)", min_value=0.0, step=1.0)
        
        Home = st.selectbox("Home Ownership", ["MORTGAGE", "RENT", "OWN", "OTHER"])
        Intent = st.selectbox("Loan Intent", ["PERSONAL", "DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "VENTURE"])

        if st.button("Predict"):
            input_data = {
                "Income": Income,
                "Emp_length": Emp_length,
                "Rate": Rate,
                "Percent_income": Percent_income,
                "Cred_length": Cred_length,
                "Home": Home,
                "Intent": Intent
            }
            
            input_df = pd.DataFrame([input_data])
            logger.info(f"Received input data: {input_df.to_dict(orient='records')[0]}")
            logger.info(f"Input DataFrame shape: {input_df.shape}")
            
            obj = PredictionPipeline()
            prediction = obj.predict(input_df)
            
            logger.info(f"Prediction result: {prediction[0]}")
            st.success(f"Predicted Credit Risk: {prediction[0]}")
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        st.error(f"Something went wrong: {str(e)}")

# Streamlit UI
def main():
    st.title("Credit Risk Prediction")
    menu = ["Home", "Train Model", "Predict"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Welcome to the Credit Risk Prediction App!")
    elif choice == "Train Model":
        train_model()
    elif choice == "Predict":
        predict_credit_risk()

if __name__ == "__main__":
    main()
