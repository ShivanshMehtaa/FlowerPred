import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

#load the pickle model;
#rb = read binary
model = pickle.load(open("model.pkl","rb"))

@app.route("/")
def HomeModel():
    return render_template("index.html")

#post bcoz we recieve independent variable value
@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)

    return render_template("index.html",prediction_text = "The Flower Species is {}".format(prediction))

if __name__ == "__main__" :
    app.run(debug = True)