from flask import Flask, render_template, request
import pickle
import os

app = Flask(
    __name__,
    template_folder="../templates",   # important
    static_folder="../static"
)

# -------------------------
# Load model safely
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "features.pkl")

model = pickle.load(open(MODEL_PATH, "rb"))
feature_names = pickle.load(open(FEATURES_PATH, "rb"))

# -------------------------
# Prepare symptoms list
# -------------------------
symptoms_list = []

for feature in feature_names:
    parts = feature.split("_")
    symptom = "_".join(parts[2:])
    symptoms_list.append(symptom.strip())

symptoms_list = sorted(list(set(symptoms_list)))

# -------------------------
# Routes
# -------------------------
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
        symptoms=symptoms_list,
        prediction=prediction
    )

# -------------------------
# Run locally
# -------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)