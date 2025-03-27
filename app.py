import os
from flask import Flask, render_template, request
import pandas as pd
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
            # Get numerical inputs
            Income = float(request.form['Income'])
            Emp_length = float(request.form['Emp_length'])
            Rate = float(request.form['Rate'])
            Percent_income = float(request.form['Percent_income'])
            Cred_length = float(request.form['Cred_length'])

            # One-hot encode categorical variables
            home_selected = request.form.get('Home', 'MORTGAGE')
            intent_selected = request.form.get('Intent', 'PERSONAL')

            # Construct input as a dictionary
            input_data = {
                "Income": Income,
                "Emp_length": Emp_length,
                "Rate": Rate,
                "Percent_income": Percent_income,
                "Cred_length": Cred_length,
                "Home": home_selected,  # Raw categorical value, let the model's encoder handle it
                "Intent": intent_selected
            }

            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])

            print('Input DataFrame shape:', input_df.shape)
            print('Input DataFrame content:', input_df)

            # Load the model and make prediction
            obj = PredictionPipeline()
            predict = obj.predict(input_df)

            return render_template('results.html', prediction=predict[0])

        except Exception as e:
            print('The Exception message is: ', e)
            return 'Something is wrong: ' + str(e)

    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(host= '0.0.0.0' , port=5000)
