#This is the script that implements the web server and the API

import numpy as np
import pandas as pd
import flask as fl
import waitress as wt
import pickle as pk

app = fl.Flask(__name__)

# Load trained model
with open("Master-Thesis/Code/app/model.pkl", "rb") as f:
    model = pk.load(f)

# Feature order must match training
MODEL_FEATURES = [
    "AgeGroup",
    "Sex",
    "DepressionType",
    "PsychoticFeatures",
    "PersonalityDisorder",
    "AnxietyDisorder",
    "InitialSetting",
    "Coercion"
]

# Dropdown options for the form
FORM_OPTIONS = {
    "AgeGroup": ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
    "Sex": ["Male", "Female"],
    "DepressionType": ["Unipolar", "Bipolar"],
    "InitialSetting": ["Inpatient", "Outpatient"],
    "Coercion": ["No", "Yes"]
}


def age_midpoint(age_group):
    """Convert age-group string to numeric midpoint."""
    if not isinstance(age_group, str):
        return np.nan
    age_group = age_group.strip()
    if "+" in age_group:
        return int(age_group.replace("+", "")) + 5
    if "-" in age_group:
        low, high = age_group.split("-")
        return (int(low) + int(high)) / 2
    return np.nan


def yes_no_to_int(value):
    """Convert Yes/No string to 1/0."""
    if isinstance(value, str):
        value = value.strip().lower()
        if value == "yes":
            return 1
        if value == "no":
            return 0
    return np.nan


def build_feature_dataframe(form_data):
    """
    Convert submitted form data into a one-row dataframe
    matching the model input schema.
    """
    features = {
        "AgeGroup": age_midpoint(form_data.get("AgeGroup")),
        "Sex": form_data.get("Sex"),
        "DepressionType": form_data.get("DepressionType"),
        "PsychoticFeatures": yes_no_to_int(form_data.get("PsychoticFeatures")),
        "PersonalityDisorder": yes_no_to_int(form_data.get("PersonalityDisorder")),
        "AnxietyDisorder": yes_no_to_int(form_data.get("AnxietyDisorder")),
        "InitialSetting": form_data.get("InitialSetting"),
        "Coercion": form_data.get("Coercion")
    }

    features_df = pd.DataFrame([features])
    features_df = features_df.loc[:, MODEL_FEATURES]
    return features_df


@app.route("/", methods=["GET"])
def home():
    return fl.render_template(
        "index.html",
        options=FORM_OPTIONS,
        prediction=None,
        probability=None,
        submitted_data={}
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        submitted_data = dict(fl.request.form)

        features_df = build_feature_dataframe(submitted_data)

        # Predict class
        prediction = model.predict(features_df)[0]

        # Predict probability if supported
        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(features_df)[0][1]) * 100

        prediction_text = (
            "Predicted remission"
            if int(prediction) == 1
            else "Predicted no remission"
        )

        return fl.render_template(
            "index.html",
            options=FORM_OPTIONS,
            prediction=prediction_text,
            probability=round(probability, 1) if probability is not None else None,
            submitted_data=submitted_data
        )

    except Exception as e:
        return fl.render_template(
            "index.html",
            options=FORM_OPTIONS,
            prediction="An error occurred while making the prediction.",
            probability=None,
            submitted_data=dict(fl.request.form),
            error=str(e)
        )


if __name__ == "__main__":
    wt.serve(app, host="0.0.0.0", port=8080)

