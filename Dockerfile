# Use an official Python image as base
FROM python:3.9

# Set the working directory
WORKDIR /loan_app

# Copy the requirements file
COPY requirements.txt .

# Debugging step: Print files inside the container
RUN ls -lah

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "loan_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
