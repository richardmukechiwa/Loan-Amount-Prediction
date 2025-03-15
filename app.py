from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline   
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from credit_risk.pipeline.prediction import PredictionPipeline

app = Flask(__name__) # initializing a flask app

@app.route('/', methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route('/train', methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 


@app.route('/predict', methods=['POST', 'GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            # Reading the inputs given by the user
            Income = float(request.form['Income'])
            Emp_length = float(request.form['Emp_length'])
            Rate = float(request.form['Rate'])
            Percent_income = float(request.form['Percent_income'])
            Cred_length = float(request.form['Cred_length'])
            
            # Home choices (one-hot encoding)
            home_options = ['MORTGAGE', 'OTHER', 'OWN', 'RENT']
            home_selected = request.form.get('Home', 'MORTGAGE')  # Default to 'MORTGAGE' if not provided
            Home = [1.0 if home_selected == h else 0.0 for h in home_options]
            
            # Intent choices (one-hot encoding)
            intent_options = ['DEBTCONSOLIDATION', 'EDUCATION', 'HOMEIMPROVEMENT', 'MEDICAL', 'PERSONAL', 'VENTURE']
            intent_selected = request.form.get('Intent', 'PERSONAL')  # Default to 'PERSONAL' if not provided
            Intent = [1.0 if intent_selected == i else 0.0 for i in intent_options]
                
            data = [Income, Emp_length, Rate, Percent_income, Cred_length] + Home + Intent
            data = np.array(data).reshape(1, 7)  # Reshape to a single sample with all features
            
            # Debug print statement to check the shape and content of the data
            print('Input data array shape:', data.shape)
            print('Input data array content:', data)
            
            obj = PredictionPipeline()
            predict = obj.predict(data)
            
            # The predict method should already scale the data correctly if the prediction is numerical
            # assume that the predict method returns the actual value and no further scaling is needed
            
            # Alternatively, if you truly need to inverse scale, you should fit the scaler on the training data of the pipeline
            # scaler = StandardScaler()
            # scaler.fit(training_data)  <- make sure to fit on the training data set of your prediction pipeline
            
            return render_template('results.html', prediction=predict[0])
        except Exception as e:
            # Print exception details for debugging
            print('The Exception message is: ', e)
            return 'Something is wrong: ' + str(e)

    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)