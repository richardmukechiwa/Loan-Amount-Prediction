from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd 
from credit_risk.pipeline.prediction import PredictionPipeline

app = Flask(__name__) #initialising a flask app

@app.route('/', methods=['GET']) #route to display the home page
def homePage():
    return render_template("index.html")

@app.route('/train', methods=['GET']) #route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!"



@app.route('/predict', methods=['POST', 'GET']) # route to show the predictions in a web UI
def index():
    if request.method== 'POST':
        try:
            # Get input values from form
            features = [
                float(request.form['Age']),
                float(request.form['Income']),
                float(request.form['Emp_length']),
                float(request.form['Rate']),
                float(request.form['Percent_income']),
                
                # Home choices (one-hot encoded)
                1.0 if request.form['Home'] == 'MORTGAGE' else 0.0,
                1.0 if request.form['Home'] == 'OTHER' else 0.0,
                1.0 if request.form['Home'] == 'OWN' else 0.0,
                1.0 if request.form['Home'] == 'RENT' else 0.0,
                
                # Intent choices (one-hot encoded)
                1.0 if request.form['Intent'] == 'DEBTCONSOLIDATION' else 0.0,
                1.0 if request.form['Intent'] == 'EDUCATION' else 0.0,
                1.0 if request.form['Intent'] == 'HOMEIMPROVEMENT' else 0.0,
                1.0 if request.form['Intent'] == 'MEDICAL' else 0.0,
                1.0 if request.form['Intent'] == 'PERSONAL' else 0.0,
                1.0 if request.form['Intent'] == 'VENTURE' else 0.0
            ]
            
            # Convert input into a NumPy array and reshape for prediction
            input_data = np.array(features).reshape(1, -1)
            prediction = model.predict(input_data)[0]
        except Exception as e:
            prediction = f"Error: {str(e)}"
            data = ["Age", "Income", "Home", "Emp_length", "Intent", "Rate", "Percent_income"]
            data =np.array(data).reshape(1, 8)
            
            obj   = PredictionPipeline()
            predict = obj.predict(data)
            
            return render_template('results.html', prediction = str(predict))
        
        except Exception as e:
            print('The Exception message is:', e)
            return 'something went wrong'
    else:
        return render_template('index.html')
    
     
            

if __name__== "__main__":
    app.run(host="0.0.0.0", port = 8080, debug=True)

