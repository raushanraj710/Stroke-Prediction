from flask import Flask, render_template, request
import numpy as np
import pickle

app= Flask(__name__)
loadmodel = pickle.load(open("stroke.pkl", "rb"))

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/result",methods=['POST','GET'])
def result():
    gender=int(request.form['gender'])
    age=int(request.form['age'])
    hypertension=int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    ever_married = int(request.form['ever_married'])
    work_type = int(request.form['work_type'])
    Residence_type = int(request.form['Residence_type'])
    avg_glucose_level = float(request.form['avg_glucose_level'])
    bmi = float(request.form['bmi'])
    smoking_status = int(request.form['smoking_status'])

   

    prediction = loadmodel.predict([[gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,
                avg_glucose_level,bmi,smoking_status]])[0]

    # for No Stroke Risk
    if prediction==0:
        return render_template('nostroke.html')
    else:
        return render_template('stroke.html')

if __name__=="__main__":
    app.run(debug=True)
