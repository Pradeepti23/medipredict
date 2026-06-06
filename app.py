from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

print("Current Folder:", os.getcwd())
print("Files:", os.listdir())

# Load Models (make sure files exist in same folder)
kidney_model = joblib.load("kidney_model.pkl")
mental_model = joblib.load("mental_model.pkl")


# Home Page
@app.route('/')
def home():
    return render_template('index.html')


# ---------------------------
# Kidney Prediction Route
# ---------------------------
@app.route('/predict_kidney', methods=['POST'])
def predict_kidney():

    age = float(request.form['age'])
    bp = float(request.form['bp'])
    sg = float(request.form['sg'])
    al = float(request.form['al'])
    su = float(request.form['su'])

    prediction = kidney_model.predict([[age, bp, sg, al, su]])
    result = "Kidney Disease Detected" if prediction[0] == 1 else "No Kidney Disease"

    return render_template('index.html', kidney_prediction=result)


# ---------------------------
# Mental Health Prediction Route
# ---------------------------
@app.route('/predict_mental', methods=['POST'])
def predict_mental():

    age = float(request.form['mental_age'])
    stress = float(request.form['stress'])
    sleep = float(request.form['sleep'])
    work = float(request.form['work'])

    #prediction = mental_model.predict([[age, stress, sleep, work]])
    prediction = mental_model.predict([[age,stress,sleep,work]]) 
    # type: ignore

    result = "Mental Health Risk Detected" if prediction[0] == 1 else "Mental Health Looks Good"

    return render_template('index.html', mental_prediction=result)


import os
print("TEMPLATE PATH:", os.path.join(os.getcwd(), "templates"))
print("FILES:", os.listdir("templates"))
if __name__ == "__main__":
    app.run(debug=True)