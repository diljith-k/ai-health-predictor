from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__, template_folder="../templates")

model = pickle.load(open("model.pkl", "rb"))
feature_names = pickle.load(open("features.pkl", "rb"))

symptoms_list = []
for feature in feature_names:
    symptom = "_".join(feature.split("_")[2:])
    symptoms_list.append(symptom.strip())

symptoms_list = sorted(list(set(symptoms_list)))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None

    if request.method == "POST":
        selected_symptoms = request.form.getlist("symptoms")
        input_vector = [0] * len(feature_names)

        for symptom in selected_symptoms:
            for i, feature in enumerate(feature_names):
                if symptom in feature:
                    input_vector[i] = 1

        prediction = model.predict([input_vector])[0]
        probs = model.predict_proba([input_vector])[0]
        confidence = round(max(probs) * 100, 2)

    return render_template(
        "index.html",
        symptoms=symptoms_list,
        prediction=prediction,
        confidence=confidence
    )