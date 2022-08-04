from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)


insurance = pickle.load(open('rfr_medical.pkl','rb'))


@application.route('/', methods=['GET'])
def Home():
    return render_template('index.html')


@application.route("/predict", methods = ["GET", "POST"])
def predict_insurance_charge():
    if request.method == 'POST':

        age = request.form['age']
        # print(Age)
        bmi = request.form['bmi']
        children = request.form['children']
        sex = request.form['sex']
        if sex == "Male":
            sex = 0

        else:
            sex = 1


        smoker = request.form['smoker']

        if smoker == "yes":
            smoker = 0


        else:
            smoker = 1


        region = request.form['region']

        if region == 'northwest':
            region = 2

        elif region == "southeast":
            region = 1

        elif region == "southwest":
            region = 0

        else:
            region = 3


        scaler = StandardScaler()
        filename_scaler = 'scaler.pkl'
        scaler_model = pickle.load(open(filename_scaler, 'rb'))
        scaled_data = scaler_model.transform([
            [age, bmi, children, sex, smoker, region]])

        prediction = insurance.predict(scaled_data)

        output = round(prediction[0], 2)
        return render_template('index.html', prediction_text="Your insurance charge is Rs. {}".format(output))

    else:
        return render_template('index.html')


if __name__ == '__main__':
    application.run(debug=True)
