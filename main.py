from flask import Flask, render_template, request
import numpy as np
import pickle




app= Flask(__name__)

model= pickle.load(open("crop_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    
     # Convert form inputs to floats
    float_features= [float(x) for x in request.form.values()]
     # Wrap in another list to make it 2D
    features= [float_features]
     # Make prediction
    prediction= model.predict(features)[0]
    
   

    return render_template("index.html",
                           prediction_text="Recommended Crop: {}".format(prediction))


if __name__=="__main__":
    app.run(debug=True)