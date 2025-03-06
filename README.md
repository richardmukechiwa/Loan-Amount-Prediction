# Loan-Amount-Prediction-Project

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

# How to run?

### STEPS:

Clone the repository

```python
https://github.com/richardmukechiwa/Loan-Amount-Prediction
```

### STEP 01- Create a conda environment after opening the repository

```python
conda create -n ln python=3.10 -y
```

```python
conda activate ln
```

### STEP 02- install the requirements

```python
pip install -r requirements.txt
```

```python
#Finally run the following command
python app.py
```

Then,
```python
open up your local host and port
```

### MLflow

[Documentation](https://mlflow.org/docs/latest/index.html)

##### cmd
- mlflow ui


### dagshub
[dagshub](https://dagshub.com/)

import dagshub
dagshub.init(repo_owner='richardmukechiwa', repo_name='Loan-Amount-Prediction', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)

Run this to export as env variables:

```python

import dagshub
import mlflow

# Initialize DagsHub MLflow tracking (automatically sets tracking URI)
dagshub.init(repo_owner='richardmukechiwa', repo_name='Loan-Amount-Prediction', mlflow=True)

# Start an MLflow experiment
with mlflow.start_run():
    mlflow.log_param('model_type', 'Linear Regression')
    mlflow.log_metric('rmse', 0.25)  # Example metric

print("MLflow tracking initialized successfully!")
```

# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 566373416292.dkr.ecr.ap-south-1.amazonaws.com/mlproj

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = simple-app




## About MLflow 
MLflow

 - Its Production Grade
 - Trace all of your expriements
 - Logging & tagging your model
