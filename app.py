from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model and features
model = pickle.load(open("model.pkl", "rb"))
feature_names = pickle.load(open("features.pkl", "rb"))

# Extract clean symptom names from feature names
symptoms_list = []

for feature in feature_names:
    parts = feature.split("_")
    symptom = "_".join(parts[2:])  # remove Symptom_1_ etc
    symptoms_list.append(symptom.strip())

# Remove duplicates and sort
symptoms_list = sorted(list(set(symptoms_list)))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        selected_symptoms = request.form.getlist("symptoms")

        # Create empty input vector
        input_vector = [0] * len(feature_names)

        # Match selected symptoms with feature names
        for symptom in selected_symptoms:
            for i, feature in enumerate(feature_names):
                if symptom in feature:
                    input_vector[i] = 1

        # Predict
        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]
        confidence = round(max(probabilities) * 100, 2)


    return render_template("index.html", 
                       prediction=prediction, 
                       confidence=confidence,
                       symptoms=symptoms)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
