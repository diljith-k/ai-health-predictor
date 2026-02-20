from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__, template_folder="../templates")

# Load model safely
model = pickle.load(open("model.pkl", "rb"))
feature_names = pickle.load(open("features.pkl", "rb"))

# Clean symptom names
symptoms_list = sorted(
    list(set("_".join(f.split("_")[2:]) for f in feature_names))
)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        selected_symptoms = request.form.getlist("symptoms")

        input_vector = [0] * len(feature_names)

        for symptom in selected_symptoms:
            for i, feature in enumerate(feature_names):
                if symptom in feature:
                    input_vector[i] = 1

        prediction = model.predict([input_vector])[0]

    return render_template(
        "index.html",
        prediction=prediction,
        symptoms=symptoms_list
    )
