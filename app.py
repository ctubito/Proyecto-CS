import os

import joblib
import pandas as pd
from flask import Flask, render_template, request

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
        try:
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
                    "Model file not found. Please train the model first "
                    "and make sure models/penguin_model.pkl exists."
                )
            else:
                input_df = pd.DataFrame([input_data])

                prediction = model.predict(input_df)[0]

                if hasattr(model, "predict_proba"):
                    probabilities_array = model.predict_proba(input_df)[0]

                    if hasattr(model, "classes_"):
                        class_names = model.classes_
                    else:
                        class_names = model.named_steps["classifier"].classes_

                    probabilities = {
                        class_name: round(probability * 100, 2)
                        for class_name, probability in zip(
                            class_names,
                            probabilities_array,
                        )
                    }

        except ValueError:
            error = "Invalid input. Please enter valid numeric values."

    return render_template(
        "predict.html",
        prediction=prediction,
        probabilities=probabilities,
        input_data=input_data,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)