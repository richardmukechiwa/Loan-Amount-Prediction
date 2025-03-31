import os
import logging
import streamlit as st
import joblib
from credit_risk.pipeline.prediction import PredictionPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load and display the image
st.image("finance.jpg", use_column_width=True)

# Train the model
def train_model():
    """Retrains the loan amount prediction model."""
    try:
        os.system("python main.py")
        logger.info("Training completed successfully.")
        st.success("Model Training Successful!")
    except Exception as e:
        logger.error("Error during training: %s", e)
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

            logger.info("Received input data: %s", input_data)

            obj = PredictionPipeline()
            predicted_loan = obj.predict(input_data)
            logger.info("Predicted Loan Amount: %s", predicted_loan[0])
            st.success(f"Estimated Loan Amount: ${predicted_loan[0]:,.2f}")

    except Exception as e:
        logger.error("Error during prediction: %s", e)
        st.error(f"Something went wrong: {str(e)}")

def main():
    """Main function to run the Streamlit app."""
    # Add a visually appealing title with a gradient background
    st.markdown(
        """
        <style>
        .main-title {
            background: linear-gradient(to right, #1f4037, #99f2c8);
            color: white;
            padding: 10px;
            border-radius: 10px;
            text-align: center;
        }
        </style>
        <div class="main-title">
            <h1>Loan Amount Prediction App</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    # Handle navigation options
    if option == "Predict Loan Amount":
        st.subheader("📊 Predict Loan Amount")
        st.write("Provide your financial details to estimate the loan amount you may qualify for.")
        predict_loan_amount()
    elif option == "Train Model":
        st.subheader("🔧 Train the Model")
        st.write("Retrain the loan prediction model with updated data.")
        train_model()
    elif option == "View Documentation":
        st.subheader("📄 Documentation")
        st.write("This app predicts the loan amount a borrower can receive based on various financial factors.")
    elif option == "View Source Code":
        st.subheader("💻 Source Code")
        st.write("Check out the source code on GitHub: [Loan Amount Prediction Repo](https://github.com/richardmukechiwa/Loan-Amount-Prediction)")
    elif option == "About":
        st.subheader("ℹ️ About the App")
        st.write("This application helps users estimate the loan amount they may qualify for based on their financial profile.")
    elif option == "Contact Us":
        st.subheader("📧 Contact Information")
        st.write("For inquiries, reach out via email at [mukechiwarichard@gmail.com](mailto:mukechiwarichard@gmail.com)")
    elif option == "Exit":
        st.write("Thank you for using the Loan Amount Prediction App!")
        
if __name__ == "__main__":
    main()
