import joblib
import pandas as pd
from flask import Flask, render_template, request

# Create a Flask app
app = Flask(__name__)

# Load the Cox model
cox_model = joblib.load('cox.pkl')

# Load the Cox-EN model
cox_en_model = joblib.load('cox-en.pkl')

# Define a route for the home page
@app.route('/')
def home():
    return render_template('homepage.html')

# Define a route for Cox model prediction
@app.route('/cox', methods=['GET', 'POST'])
def predict_cox():
    if request.method == 'POST':
        # Get user input from the form
        age = float(request.form['age'])
        sex = int(request.form['sex'])
        dm = int(request.form['dm'])
        who = int(request.form['who'])
        gcs = int(request.form['gcs'])
        nihss = int(request.form['nihss'])
        mrs = int(request.form['mrs'])

        # Create an input DataFrame for prediction
        input_data = pd.DataFrame(
            {'age': [age], 'sex': [sex], 'dm': [dm],
             'who': [who], 'gcs': [gcs],
             'nihss': [nihss], 'mrs': [mrs]})

        # Use the Cox model to make a prediction
        prediction = cox_model.predict_survival_function(input_data)

        # Convert the prediction to HTML (as a string)
        prediction_html = prediction.to_html()

        return render_template('cox.html', cox_prediction=prediction_html, cox_en_prediction=None)

    return render_template('cox.html', cox_prediction=None, cox_en_prediction=None)

# Define a route for Cox-EN model prediction
@app.route('/cox-en', methods=['GET', 'POST'])
def predict_cox_en():
    if request.method == 'POST':
        # Get user input from the form
        age = float(request.form['age'])
        sex = int(request.form['sex'])
        dm = int(request.form['dm'])
        who = int(request.form['who'])
        gcs = int(request.form['gcs'])
        nihss = int(request.form['nihss'])
        mrs = int(request.form['mrs'])

        # Create an input DataFrame for prediction
        input_data = pd.DataFrame(
            {'age': [age], 'sex': [sex], 'dm': [dm],
             'who': [who], 'gcs': [gcs],
             'nihss': [nihss], 'mrs': [mrs]})

        # Use the Cox-EN model to make a prediction
        prediction = cox_en_model.predict_survival_function(input_data)

        # Convert the prediction to HTML (as a string)
        prediction_html = prediction.to_html()

        return render_template('cox-en.html', cox_prediction=None, cox_en_prediction=prediction_html)

    return render_template('cox-en.html', cox_prediction=None, cox_en_prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
