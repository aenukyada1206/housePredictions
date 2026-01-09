from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load ML model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# 🔹 Render HTML page
@app.route("/")
def home():
    return render_template("index.html")

# 🔹 Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    area = data.get("area")
    bedrooms = data.get("bedrooms")

    if area is None or bedrooms is None:
        return jsonify({"error": "area and bedrooms are required"}), 400

    features = np.array([[area, bedrooms]])
    prediction = model.predict(features)

    return jsonify({
        "predicted_price": round(float(prediction[0]), 2)
    })

# 🔹 Local run
if __name__ == "__main__":
    # app.run(debug=True)
    pass
