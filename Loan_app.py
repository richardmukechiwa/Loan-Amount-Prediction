import os
import logging
import streamlit as st
import pandas as pd
from loan_amount.pipeline.prediction import PredictionPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to train the model
def train_model():
    st.subheader("Train the Loan Amount Prediction Model")
    try:
        os.system("python main.py")
        logger.info("Training completed successfully.")
        st.success("Training Successful! 🎉")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        st.error(f"Training failed: {str(e)}")

# Function to predict loan amount
def predict_loan_amount():
    st.subheader("Enter Borrower Details for Loan Amount Prediction")

    # User Inputs
    Income = st.number_input("Income ($)", min_value=0.0, step=100.0)
    Emp_length = st.number_input("Employment Length (Years)", min_value=0.0, step=1.0)
    Credit_score = st.number_input("Credit Score", min_value=300, max_value=850, step=1)
    Debt_to_income_ratio = st.number_input("Debt-to-Income Ratio (%)", min_value=0.0, step=0.1)
    Existing_loans = st.number_input("Number of Existing Loans", min_value=0, step=1)

    Home = st.selectbox("Home Ownership", ["MORTGAGE", "RENT", "OWN", "OTHER"])
    Loan_purpose = st.selectbox("Loan Purpose", ["PERSONAL", "DEBT CONSOLIDATION", "EDUCATION", "HOME IMPROVEMENT", "MEDICAL", "BUSINESS"])

    if st.button("Predict Loan Amount"):
        input_data = {
            "Income": Income,
            "Emp_length": Emp_length,
            "Credit_score": Credit_score,
            "Debt_to_income_ratio": Debt_to_income_ratio,
            "Existing_loans": Existing_loans,
            "Home": Home,
            "Loan_purpose": Loan_purpose
        }

        input_df = pd.DataFrame([input_data])
        logger.info(f"Received input data: {input_df.to_dict(orient='records')[0]}")
        logger.info(f"Input DataFrame shape: {input_df.shape}")

        try:
            obj = PredictionPipeline()
            predicted_amount = obj.predict(input_df)
            logger.info(f"Predicted Loan Amount: ${predicted_amount[0]:,.2f}")
            st.success(f"Predicted Loan Amount: ${predicted_amount[0]:,.2f}")
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            st.error(f"Something went wrong: {str(e)}")

# Function to show documentation
def view_documentation():
    st.subheader("📄 Documentation")
    st.write("This app predicts the maximum loan amount a borrower can qualify for based on financial and personal factors.")
    st.markdown("""
    ### **How It Works**
    - The model analyzes the borrower's income, credit score, and other factors.
    - It predicts the optimal loan amount the borrower can receive.
    - The model is trained on real-world loan data to ensure accuracy.

    **Key Features:**
    - Predict loan amount eligibility.
    - Train the model with updated data.
    - View source code for transparency.
    """)

# Function to show source code link
def view_source_code():
    st.subheader("🔗 Source Code")
    st.write("The source code is available on GitHub:")
    st.markdown("[View on GitHub](https://github.com/richardmukechiwa/Loan-Amount-Prediction)", unsafe_allow_html=True)

# Function to display About information
def about_app():
    st.subheader("ℹ️ About This App")
    st.write("""
    This application helps borrowers estimate the loan amount they can receive based on their financial profile.
    It uses machine learning to provide accurate predictions, assisting users in making informed financial decisions.
    """)

# Function to display Contact information
def contact_us():
    st.subheader("📬 Contact Us")
    st.write("For inquiries, feel free to reach out:")
    st.write("📧 Email: [mukechiwarichard@gmail.com](mailto:mukechiwarichard@gmail.com)")
    st.write("💼 LinkedIn: [Richard Mukechiwa](https://www.linkedin.com/in/richardmukechiwa)")

# Function to exit the app
def exit_app():
    st.warning("🔴 Exiting the application...")
    st.stop()

# Main function for UI
def main():
    st.title("📊 Loan Amount Prediction App")
    
    html_temp = """
    <div style="background-color:blue;padding:10px">
    <h2 style="color:white;text-align:center;">Loan Amount Prediction</h2>  
    </div>  
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Sidebar Navigation
    menu = {
        "Predict Loan Amount": predict_loan_amount,
        "Train Model": train_model,
        "View Documentation": view_documentation,
        "View Source Code": view_source_code,
        "About": about_app,
        "Contact Us": contact_us,
        "Exit": exit_app
    }

    choice = st.sidebar.radio("🔍 Navigation", list(menu.keys()))
    menu[choice]()  # Execute selected function

if __name__ == "__main__":
    main()
