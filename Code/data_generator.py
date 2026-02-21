import pandas as pd
import numpy as np

# Number of samples
n = 2000

# Generate IDs
ids = np.arange(1, n+1)

# Age distribution (mean ~58.8, SD ~15)
ages = np.random.normal(58.8, 15, n).clip(18, 90).round(1)

# Sex distribution (76.7% female)
sex = np.random.choice(["Female", "Male"], size=n, p=[0.767, 0.233])

# Baseline HDRS (mean 24.8 ± 6.0)
baseline_hdrs = np.random.normal(24.8, 6.0, n).clip(10, 40).round(1)

# Endpoint HDRS (mean 8.5 ± 5.0)
endpoint_hdrs = np.random.normal(8.5, 5.0, n).clip(0, 30).round(1)

# Average HDRS decrease % (mean 63.9 ± 22.9)
hdrs_decrease = np.random.normal(63.9, 22.9, n).clip(0, 100).round(1)

# Responders (73.9%)
responders = np.random.choice([1, 0], size=n, p=[0.739, 0.261])

# Remitters (56.2%)
remitters = np.random.choice([1, 0], size=n, p=[0.562, 0.438])

# CORE-defined melancholia (63%)
melancholia = np.random.choice([1, 0], size=n, p=[0.63, 0.37])

# Bipolar (17.8%)
bipolar = np.random.choice([1, 0], size=n, p=[0.178, 0.822])

# Episode duration mean 14.3 ± 18.1 (median 6.5, range 1–84)
episode_duration = np.random.normal(14.3, 18.1, n).clip(1, 84).round(1)

# Maudsley staging method - episode duration categories
episode_stage = np.random.choice(
    ["Acute (<12m)", "Subacute (12-24m)", "Chronic (>24m)"], 
    size=n, p=[48/73, 12/73, 13/73]
)

# Failed treatments distribution
probs = np.array([22, 29, 13, 5])
probs = probs / probs.sum()  # normalize
failed_treatments = np.random.choice(
    ["1-2", "3-4", "5-6", "7-10"],
    size=n,
    p=probs
)

# Depression severity
severity = np.random.choice(
    ["Moderate (HDRS 17-23)", 
     "Severe no psychosis (HDRS ≥24)", 
     "Severe with psychosis (HDRS ≥24)"], 
    size=n, p=[25/73, 17/73, 31/73]
)

# Augmentation (90.4%)
augmentation = np.random.choice([1, 0], size=n, p=[0.904, 0.096])

# Previous electroconvulsive therapy (2.7%)
ect = np.random.choice([1, 0], size=n, p=[0.027, 0.973])

# Build DataFrame
df = pd.DataFrame({
    "ID": ids,
    "Age": ages,
    "Sex": sex,
    "Baseline_HDRS": baseline_hdrs,
    "Endpoint_HDRS": endpoint_hdrs,
    "HDRS_Decrease_%": hdrs_decrease,
    "Responder": responders,
    "Remitter": remitters,
    "Melancholia": melancholia,
    "Bipolar": bipolar,
    "Episode_Duration_Months": episode_duration,
    "Episode_Stage": episode_stage,
    "Failed_Treatments": failed_treatments,
    "Severity": severity,
    "Augmentation": augmentation,
    "ECT_Previous": ect
})

# Save to CSV
df.to_csv("study_population_dataset.csv", index=False)
print("CSV file 'study_population_dataset.csv' has been created with", n, "rows.")
