#This is the script that implements the web server and the API

import numpy as np
import pandas as pd
import flask as fl
import waitress as wt
import pickle as pk

app = fl.Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as f:
    model = pk.load(f)

# Feature order must match training
MODEL_FEATURES = [
    "Sex",
    "AnxietyDisorder",
    "PsychoticFeatures",
    "Education",
    "AgeGroup",
    "SubstanceUse",
    "MaritalStatus",
    "DepressionType"
]

FORM_OPTIONS = {
    "AgeGroup": ["18-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-95"],
    "Sex": ["Female", "Male"],
    "DepressionType": ["Unipolar", "Bipolar"],
    "Education": ["≤9 years", "10-12 years", ">12 years"],
    "MaritalStatus": ["Unmarried", "Married", "Divorced", "Widowed"]
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
    if value is None:
        return np.nan
    value = str(value).strip().lower()
    if value in {"yes", "1"}:
        return 1
    if value in {"no", "0"}:
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
        "AnxietyDisorder": yes_no_to_int(form_data.get("AnxietyDisorder")),
        "SubstanceUse": yes_no_to_int(form_data.get("SubstanceUse")),
        "Education": form_data.get("Education"),
        "MaritalStatus": form_data.get("MaritalStatus")
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

        probability_rounded = round(probability, 1) if probability is not None else None
        icon_count = int(round(probability)) if probability is not None else None

        return fl.render_template(
            "index.html",
            options=FORM_OPTIONS,
            prediction=prediction_text,
            probability=probability_rounded,
            icon_count=icon_count,
            submitted_data=submitted_data
        )

    except Exception as e:
        return fl.render_template(
            "index.html",
            options=FORM_OPTIONS,
            prediction="An error occurred while making the prediction.",
            probability=None,
            icon_count=None,
            submitted_data=dict(fl.request.form),
            error=str(e)
        )


if __name__ == "__main__":
    wt.serve(app, host="0.0.0.0", port=8080)

