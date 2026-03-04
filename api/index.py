from flask import Flask, render_template, request

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

# Sample symptoms list
symptoms_list = [
    "Fever",
    "Headache",
    "Cough",
    "Fatigue",
    "Nausea"
]

@app.route("/")
def home():
    return render_template("index.html", symptoms=symptoms_list)

@app.route("/predict", methods=["POST"])
def predict():
    selected = request.form.getlist("symptoms")
    result = "You selected: " + ", ".join(selected) if selected else "No symptoms selected"
    return render_template("index.html", symptoms=symptoms_list, result=result)

# Required for Vercel
app = app