# Use an official Python image as base
FROM python:3.9

# Set the working directory inside the container
WORKDIR /loan_app

# Copy the requirements file first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Expose the Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "loan_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
