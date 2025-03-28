import os
import logging
import streamlit as st
import pandas as pd
from credit_risk.pipeline.prediction import PredictionPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model():
    """Retrains the loan amount prediction model."""
    try:
        os.system("python main.py")
        logger.info("Training completed successfully.")
        st.success("Model Training Successful!")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        st.error(f"Training failed: {str(e)}")

def predict_loan_amount():
    """Takes user input and predicts the loan amount they can receive."""
    try:
        st.subheader("Enter Details for Loan Amount Prediction")
        
        # Numerical Inputs
        Income = st.number_input("Income ($)", min_value=0.0, step=100.0)
        Emp_length = st.number_input("Employment Length (Years)", min_value=0.0, step=1.0)
        Rate = st.number_input("Interest Rate (%)", min_value=0.0, step=0.1)
        Percent_income = st.number_input("Percent of Income for Loan (%)", min_value=0.0, step=0.1)
        Cred_length = st.number_input("Credit Length (Years)", min_value=0.0, step=1.0)
        
        # Categorical Inputs
        Home = st.selectbox("Home Ownership", ["MORTGAGE", "RENT", "OWN", "OTHER"])
        Intent = st.selectbox("Loan Intent", ["PERSONAL", "DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "VENTURE"])
        
        if st.button("Predict Loan Amount"):
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
            
            obj = PredictionPipeline()
            predicted_loan = obj.predict(input_df)
            
            logger.info(f"Predicted Loan Amount: {predicted_loan[0]}")
            st.success(f"Estimated Loan Amount: ${predicted_loan[0]:,.2f}")
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        st.error(f"Something went wrong: {str(e)}")

def main():
    """Main function to run the Streamlit app."""
    st.title("Loan Amount Prediction App")
    
    html_temp = """
    <div style="background-color:blue;padding:10px">
    <h2 style="color:white;text-align:center;">Loan Amount Prediction</h2>  
    </div>  
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    option = st.sidebar.radio("Select an option:", [
        "Predict Loan Amount",
        "Train Model",
        "View Documentation",
        "View Source Code",
        "About",
        "Contact Us",
        "Exit"
    ])
    
    if option == "Predict Loan Amount":
        predict_loan_amount()
    elif option == "Train Model":
        train_model()
    elif option == "View Documentation":
        st.subheader("Documentation")
        st.write("This app predicts the loan amount a borrower can receive based on various financial factors.")
    elif option == "View Source Code":
        st.subheader("Source Code")
        st.write("Check out the source code on GitHub: [Loan Amount Prediction Repo](https://github.com/richardmukechiwa/Loan-Amount-Prediction)")
    elif option == "About":
        st.subheader("About the App")
        st.write("This application helps users estimate the loan amount they may qualify for based on their financial profile.")
    elif option == "Contact Us":
        st.subheader("Contact Information")
        st.write("For inquiries, reach out via email at [mukechiwarichard@gmail.com](mailto:mukechiwarichard@gmail.com)")
    elif option == "Exit":
        st.write("Thank you for using the Loan Amount Prediction App!")
    
if __name__ == "__main__":
    main()
