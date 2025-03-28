import os
import logging
import streamlit as st
import pandas as pd
from credit_risk.pipeline.prediction import PredictionPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to train the model
def train_model():
    st.subheader("Train the Credit Risk Model")
    try:
        os.system("python main.py")
        logger.info("Training completed successfully.")
        st.success("Training Successful! 🎉")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        st.error(f"Training failed: {str(e)}")

# Function to predict credit risk
def predict_credit_risk():
    st.subheader("Enter Loan Applicant Details for Prediction")

    # User Inputs
    Income = st.number_input("Income ($)", min_value=0.0, step=100.0)
    Emp_length = st.number_input("Employment Length (Years)", min_value=0.0, step=1.0)
    Rate = st.number_input("Interest Rate (%)", min_value=0.0, step=0.1)
    Percent_income = st.number_input("Percent of Income", min_value=0.0, step=0.1)
    Cred_length = st.number_input("Credit Length (Years)", min_value=0.0, step=1.0)

    Home = st.selectbox("Home Ownership", ["MORTGAGE", "RENT", "OWN", "OTHER"])
    Intent = st.selectbox("Loan Intent", ["PERSONAL", "DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "VENTURE"])

    if st.button("Predict Risk"):
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

        try:
            obj = PredictionPipeline()
            prediction = obj.predict(input_df)
            logger.info(f"Prediction result: {prediction[0]}")
            st.success(f"Predicted Credit Risk: {prediction[0]}")
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            st.error(f"Something went wrong: {str(e)}")

# Function to show documentation
def view_documentation():
    st.subheader("📄 Documentation")
    st.write("This app predicts the risk level of a loan applicant using machine learning models.")
    st.markdown("""
    ### **How It Works**
    - The model takes various financial and personal attributes as input.
    - It predicts the likelihood of loan repayment risk.
    - The model is trained on historical loan data.

    **Key Features:**
    - Predict credit risk based on applicant details.
    - Train the model using updated data.
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
    This application helps financial institutions assess the risk associated with a loan applicant.
    It utilizes machine learning to predict whether an applicant is a high or low credit risk.
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
    st.title("📊 Credit Risk Prediction App")
    
    html_temp = """
    <div style="background-color:blue;padding:10px">
    <h2 style="color:white;text-align:center;">Credit Risk Prediction</h2>  
    </div>  
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Sidebar Navigation
    menu = {
        "Predict Credit Risk": predict_credit_risk,
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
