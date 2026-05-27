import os

from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model_path = os.path.join(app.root_path, "models", "penguin_model.pkl")
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    probabilities = None
    input_data = None
    error = None

    if request.method == "POST":
        input_data = {
            "bill_length_mm": float(request.form["bill_length_mm"]),
            "bill_depth_mm": float(request.form["bill_depth_mm"]),
            "flipper_length_mm": float(request.form["flipper_length_mm"]),
            "body_mass_g": float(request.form["body_mass_g"]),
            "island": request.form["island"],
            "sex": request.form["sex"],
        }

        if model is None:
            error = (
                "Model file not found. Please add `models/penguin_model.pkl` "
                "or train and save the model before using prediction."
            )
        else:
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]

            if hasattr(model, "predict_proba"):
                probabilities_array = model.predict_proba(input_df)[0]
                class_names = model.classes_

                probabilities = {
                str(class_name): float(round(probability * 100, 2))
        prediction=prediction,
        probabilities=probabilities,
        input_data=input_data,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)