from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = pickle.load(open("crop_model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    crop = None
    if request.method == "POST":
        # Get form data
        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        # Prepare the feature array for prediction
        features = np.array([N, P, K, temperature, humidity, ph, rainfall]).reshape(1, -1)

        # Make prediction
        crop = model.predict(features)[0]
    
    # Render the HTML page
    return render_template("index.html", crop=crop)

if __name__ == "__main__":
    app.run(debug=True)
