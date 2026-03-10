import numpy as np
import pandas as pd

np.random.seed(42)

n = 8000

# --- Helper functions ---
def categorical(probabilities, labels):
    return np.random.choice(labels, size=n, p=probabilities)

def binary(prob):
    return np.random.binomial(1, prob, size=n)

# --- Variables based on proportions roughly from the table ---

sex = categorical([0.55, 0.45], ["Female", "Male"])
age_group = categorical(
    [0.1, 0.15, 0.25, 0.25, 0.15, 0.07, 0.03],
    ["18-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90"]
)
marital_status = categorical([0.35, 0.45, 0.10, 0.10],
                             ["Married", "Unmarried", "Widowed", "Divorced"])
education = categorical([0.25, 0.45, 0.30],
                        ["≤10 years", "10-12 years", ">12 years"])
depression_type = categorical([0.75, 0.25],
                              ["Unipolar", "Bipolar"])
psychotic_features = binary(0.45)
substance_use = binary(0.15)
personality_disorder = binary(0.25)
anxiety_disorder = binary(0.50)
antidepressant_before_ECT = categorical([0.28, 0.25, 0.25, 0.22],
                                        ["Never", "0-3 months before", "0-3 months after", "≥6 months after"])
initial_setting = categorical([0.70, 0.30],
                              ["Inpatient", "Outpatient"])
coercion = categorical([0.80, 0.20],
                       ["Voluntary", "Involuntary"])
antidepressant_med = binary(0.70)
lithium = binary(0.10)
lamotrigine = binary(0.05)
valproate = binary(0.15)
benzodiazepines = binary(0.45)
antipsychotic_med = binary(0.55)

# --- Generate remission outcome (roughly matching odds ratios) ---
# Base probability
base_p = 0.45

# Add some effects (higher or lower remission odds)
remission_prob = (
    base_p
    + 0.05 * (sex == "Male")
    - 0.10 * (age_group == "81-90")
    - 0.05 * (personality_disorder == 1)
    - 0.08 * (psychotic_features == 1)
    - 0.05 * (substance_use == 1)
    + 0.10 * (depression_type == "Bipolar")
    + 0.08 * (lithium == 1)
    - 0.07 * (lamotrigine == 1)
    - 0.05 * (valproate == 1)
)

remission_prob = np.clip(remission_prob, 0.05, 0.95)
remission = np.random.binomial(1, remission_prob)

# --- Combine into DataFrame ---
data = pd.DataFrame({
    "Sex": sex,
    "AgeGroup": age_group,
    "MaritalStatus": marital_status,
    "Education": education,
    "DepressionType": depression_type,
    "PsychoticFeatures": psychotic_features,
    "SubstanceUse": substance_use,
    "PersonalityDisorder": personality_disorder,
    "AnxietyDisorder": anxiety_disorder,
    "AntidepressantBeforeECT": antidepressant_before_ECT,
    "InitialSetting": initial_setting,
    "Coercion": coercion,
    "AntidepressantMedication": antidepressant_med,
    "Lithium": lithium,
    "Lamotrigine": lamotrigine,
    "Valproate": valproate,
    "Benzodiazepines": benzodiazepines,
    "AntipsychoticMedication": antipsychotic_med,
    "Remission": remission
})

# --- Save to CSV ---
data.to_csv("ect_remission_data.csv", index=False)
print("✅ Dataset created: ect_remission_data.csv")
print(data.head())
