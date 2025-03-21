from pathlib import Path
from flask import Flask, render_template, request
import numpy as np
import os
from credit_risk.pipeline.prediction import PredictionPipeline

app = Flask(__name__)

@app.route('/', methods=['GET'])
def homePage():
    return render_template("index.html")

@app.route('/train', methods=['GET'])
def training():
    os.system("python main.py")
    return "Training Successful!"

@app.route('/predict', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        try:
            Income = float(request.form['Income'])
            Emp_length = float(request.form['Emp_length'])
            Rate = float(request.form['Rate'])
            Percent_income = float(request.form['Percent_income'])
            Cred_length = float(request.form['Cred_length'])
            
            home_options = ['MORTGAGE', 'OTHER', 'OWN', 'RENT']
            home_selected = request.form.get('Home', 'MORTGAGE')
            Home = [1.0 if home_selected == h else 0.0 for h in home_options]
            
            intent_options = ['DEBTCONSOLIDATION', 'EDUCATION', 'HOMEIMPROVEMENT', 'MEDICAL', 'PERSONAL', 'VENTURE']
            intent_selected = request.form.get('Intent', 'PERSONAL')
            Intent = [1.0 if intent_selected == i else 0.0 for i in intent_options]
            
            data = [Income, Emp_length, Rate, Percent_income, Cred_length] + Home + Intent
            data = np.array(data).reshape(1, 15)
            
            print('Input data array shape:', data.shape)
            print('Input data array content:', data)
            
            obj = PredictionPipeline()
            predict = obj.predict(data)
            
            return render_template('results.html', prediction=predict[0])
        except Exception as e:
            print('The Exception message is: ', e)
            return 'Something is wrong: ' + str(e)
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)