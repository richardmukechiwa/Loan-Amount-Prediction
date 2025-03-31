import os
import logging
import streamlit as st
import plotly.express as px
from streamlit_option_menu import option_menu
from credit_risk.pipeline.prediction import PredictionPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Train Model Function
def train_model():
    try:
        os.system("python main.py")
        logger.info("Training completed successfully.")
        st.success("Model Training Successful!")
    except Exception as e:
        logger.error("Error during training: %s", e)
        st.error(f"Training failed: {str(e)}")

# Loan Prediction Function
def predict_loan_amount():
    st.subheader("📊 Enter Your Details")

    col1, col2 = st.columns(2)
    
    with col1:
        Income = st.number_input("Income ($)", min_value=0.0, step=100.0)
        Emp_length = st.number_input("Employment Length (Years)", min_value=0.0, step=1.0)
        Rate = st.number_input("Interest Rate (%)", min_value=0.0, step=0.1)
    
    with col2:
        Percent_income = st.number_input("% of Income for Loan", min_value=0.0, step=0.1)
        Cred_length = st.number_input("Credit Length (Years)", min_value=0.0, step=1.0)
        Home = st.selectbox("Home Ownership", ["MORTGAGE", "RENT", "OWN", "OTHER"])
    
    Intent = st.selectbox("Loan Intent", ["PERSONAL", "DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "VENTURE"])

    if st.button("💰 Predict Loan Amount"):
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
        
        # Loan Prediction Visualization
        fig = px.bar(x=["Minimum Loan", "Predicted Loan", "Maximum Loan"],
                     y=[predicted_loan[0]*0.8, predicted_loan[0], predicted_loan[0]*1.2],
                     title="Estimated Loan Amount Range", color_discrete_sequence=["#1f77b4"])
        
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"Estimated Loan Amount: ${predicted_loan[0]:,.2f}")

# Main App Layout
def main():
    # Header with background image
    st.image("finance.jpg", use_container_width=True)
    st.markdown("""
    <h1 style='text-align: center; color: #1f4037;'>Loan Amount Prediction App</h1>
    """, unsafe_allow_html=True)

    # Sidebar Navigation
    with st.sidebar:
        selected = option_menu(
            "Navigation", ["🏠 Home", "📊 Predict Loan", "🔧 Train Model", "📄 Documentation", "💻 Source Code", "ℹ️ About", "📧 Contact"],
            icons=["house", "graph-up", "wrench", "file-earmark-text", "code", "info-circle", "envelope"],
            menu_icon="menu-up", default_index=1
        )

    if selected == "🏠 Home":
        st.subheader("Welcome to the Loan Prediction App!")
        st.write("Use this tool to estimate the loan amount you may qualify for based on your financial profile.")
    elif selected == "📊 Predict Loan":
        predict_loan_amount()
    elif selected == "🔧 Train Model":
        train_model()
    elif selected == "📄 Documentation":
        st.subheader("📄 Documentation")
        st.write("This app predicts the loan amount based on user-provided financial details.")
    elif selected == "💻 Source Code":
        st.subheader("💻 Source Code")
        st.write("Check out the source code on GitHub: [Loan Amount Prediction Repo](https://github.com/richardmukechiwa/Loan-Amount-Prediction)")
    elif selected == "ℹ️ About":
        st.subheader("ℹ️ About the App")
        st.write("This application helps users estimate the loan amount they may qualify for based on their financial profile.")
    elif selected == "📧 Contact":
        st.subheader("📧 Contact Information")
        st.write("For inquiries, reach out via email at [mukechiwarichard@gmail.com](mailto:mukechiwarichard@gmail.com)")

if __name__ == "__main__":
    main()

