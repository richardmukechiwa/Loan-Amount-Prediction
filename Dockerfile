# Use the official Python image as the base
FROM python:3.9

# Set the working directory inside the container
WORKDIR /loan_app

# Copy only requirements.txt first
COPY requirements.txt /app/requirements.txt

# Upgrade pip (recommended)
RUN pip install --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application code
COPY . /loan_app

# Expose the Streamlit default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "loan_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
