from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
feature_names = pickle.load(open(os.path.join(BASE_DIR, "features.pkl"), "rb"))


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        selected_symptoms = data.get("symptoms", [])

        input_vector = [0] * len(feature_names)

        for symptom in selected_symptoms:
            if symptom in feature_names:
                index = feature_names.index(symptom)
                input_vector[index] = 1

        prediction = model.predict([input_vector])[0]

        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = max(model.predict_proba([input_vector])[0])

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence * 100, 2) if confidence else None
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/symptoms", methods=["GET"])
def get_symptoms():
    return jsonify({
        "symptoms": feature_names
    })


@app.route("/")
def home():
    return "Flask API running"


if __name__ == "__main__":
    app.run(port=5050)