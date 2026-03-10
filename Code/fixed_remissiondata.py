import numpy as np
import pandas as pd

np.random.seed(42)

# -----------------------------
# Dataset size
# -----------------------------
n = 8000

# -----------------------------
# Helper functions
# -----------------------------
def categorical(probabilities, labels, size=n):
    return np.random.choice(labels, size=size, p=probabilities)

def binary(prob, size=n):
    return np.random.binomial(1, prob, size=size)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calibrate_intercept(eta, target_rate, low=-10, high=10, tol=1e-10, max_iter=200):
    """
    Find intercept b0 such that mean(sigmoid(b0 + eta)) ~= target_rate
    """
    for _ in range(max_iter):
        mid = (low + high) / 2
        rate = sigmoid(mid + eta).mean()
        if abs(rate - target_rate) < tol:
            return mid
        if rate < target_rate:
            low = mid
        else:
            high = mid
    return (low + high) / 2

# -----------------------------
# Proportions from full counts in the table
# denominator = 1671
# -----------------------------

# Sex: Female 588+433=1021, Male 367+283=650
sex = categorical(
    [1021/1671, 650/1671],
    ["Female", "Male"]
)

# Age groups:
# 18-30: 142+59=201
# 31-40: 139+62=201
# 41-50: 207+101=308
# 51-60: 195+122=317
# 61-70: 151+179=330
# 71-80: 89+116=205
# 81-95: 32+77=109
age_group = categorical(
    [201/1671, 201/1671, 308/1671, 317/1671, 330/1671, 205/1671, 109/1671],
    ["18-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-95"]
)

# Marital status:
# Married 324+288=612
# Unmarried 372+208=580
# Widowed 53+77=130
# Divorced 206+143=349
marital_status = categorical(
    [612/1671, 580/1671, 130/1671, 349/1671],
    ["Married", "Unmarried", "Widowed", "Divorced"]
)

# Education:
# <=9 years or missing: 207+147=354
# 10-12 years: 412+344=756
# >12 years: 336+225=561
education = categorical(
    [354/1671, 756/1671, 561/1671],
    ["≤9 years", "10-12 years", ">12 years"]
)

# Depression type:
# Unipolar 735+599=1334
# Bipolar 220+117=337
depression_type = categorical(
    [1334/1671, 337/1671],
    ["Unipolar", "Bipolar"]
)

# Binary variables from full subgroup totals
# Psychotic features: Yes 107+188=295
psychotic_features = binary(295/1671)

# Substance use disorders: Yes 248+101=349
substance_use = binary(349/1671)

# Personality disorder: Yes 158+55=213
personality_disorder = binary(213/1671)

# Anxiety disorder: Yes 364+182=546
anxiety_disorder = binary(546/1671)

# Initial treatment setting: Outpatient 152+75=227
initial_setting = categorical(
    [1444/1671, 227/1671],
    ["Inpatient", "Outpatient"]
)

# Coercion: Involuntary hospitalization 75+100=175
coercion = categorical(
    [1496/1671, 175/1671],
    ["Voluntary", "Involuntary"]
)

# Antidepressant medication: Yes 836+619=1455
antidepressant_med = binary(1455/1671)

# Lithium: Yes 175+94=269
lithium = binary(269/1671)

# Lamotrigine: Yes 131+36=167
lamotrigine = binary(167/1671)

# Valproate: Yes 32+13=45
valproate = binary(45/1671)

# -----------------------------
# Logistic model using adjusted ORs
# Reference categories:
# Female, 18-30, Married, <=9 years, Unipolar, no psychotic,
# no substance use, no personality disorder, no anxiety,
# Inpatient, Voluntary, No antidepressant medication, No lithium,
# No lamotrigine, No valproate
# -----------------------------

eta = np.zeros(n)

# Sex
eta += np.log(0.97) * (sex == "Male")

# Age
eta += np.log(1.12) * (age_group == "31-40")
eta += np.log(1.13) * (age_group == "41-50")
eta += np.log(1.27) * (age_group == "51-60")
eta += np.log(2.09) * (age_group == "61-70")
eta += np.log(2.18) * (age_group == "71-80")
eta += np.log(5.03) * (age_group == "81-95")

# Marital status
eta += np.log(0.98) * (marital_status == "Unmarried")
eta += np.log(0.86) * (marital_status == "Widowed")
eta += np.log(0.87) * (marital_status == "Divorced")

# Education
eta += np.log(1.59) * (education == "10-12 years")
eta += np.log(1.27) * (education == ">12 years")

# Depression diagnosis
eta += np.log(1.10) * (depression_type == "Bipolar")

# Clinical variables
eta += np.log(1.94) * psychotic_features
eta += np.log(0.74) * substance_use
eta += np.log(0.75) * personality_disorder
eta += np.log(0.72) * anxiety_disorder

# Initial setting
eta += np.log(0.85) * (initial_setting == "Outpatient")

# Coercion
eta += np.log(1.39) * (coercion == "Involuntary")

# Medications
eta += np.log(0.75) * antidepressant_med   # Yes vs No
eta += np.log(0.74) * lithium              # Yes vs No
eta += np.log(0.48) * lamotrigine          # Yes vs No
eta += np.log(0.58) * valproate            # Yes vs No

# -----------------------------
# Calibrate intercept to overall remission rate
# Table total remission = 433 + 283 = 716
# Total n = 1671
# -----------------------------
target_remission_rate = 716 / 1671

intercept = calibrate_intercept(eta, target_remission_rate)
p = sigmoid(intercept + eta)

# Generate remission outcome
remission = np.random.binomial(1, p)

# -----------------------------
# Build DataFrame
# -----------------------------
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
    "InitialSetting": initial_setting,
    "Coercion": coercion,
    "AntidepressantMedication": antidepressant_med,
    "Lithium": lithium,
    "Lamotrigine": lamotrigine,
    "Valproate": valproate,
    "Remission": remission
})

# Save
data.to_csv("ect_remission_synthetic.csv", index=False)

print("Dataset created: ect_remission_synthetic.csv")
print("Target remission rate:", round(target_remission_rate, 4))
print("Observed remission rate:", round(data["Remission"].mean(), 4))
print("\nColumn proportions:\n")
print("Sex:\n", data["Sex"].value_counts(normalize=True).round(3))
print("\nAntidepressantMedication:\n", data["AntidepressantMedication"].value_counts(normalize=True).round(3))
print("\nHead:\n", data.head())