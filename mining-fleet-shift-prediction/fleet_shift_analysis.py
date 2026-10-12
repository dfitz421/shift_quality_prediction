from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


# The repo includes a synthetic sample so the analysis can be run without
# distributing the original source dataset.
DATA_FILE = Path("sample_shift_data.csv")

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)
print("\nExample rows:")
print(df[[
    "Production/bcm",
    "Queue load duration/s",
    "Queue dump duration/s",
    "Off circuit travel time/s",
    "Number of trucks",
    "Truck available hours",
]].head())


# -------------------------------------------------------------------
# 1. Define shift performance
# -------------------------------------------------------------------

# Primary KPI: production normalized by the amount of truck capacity
# available during the shift.
df["productivity"] = (
    df["Production/bcm"] / df["Truck available hours"]
)

# Two timing measures I wanted to compare with productivity.
df["total_queue_time"] = (
    df["Queue load duration/s"] + df["Queue dump duration/s"]
)
df["off_circuit_time"] = df["Off circuit travel time/s"]


# There were no business-defined cutoffs for a good or bad shift, so I
# divided the productivity distribution into thirds.
low_cutoff = df["productivity"].quantile(0.33)
high_cutoff = df["productivity"].quantile(0.66)


def label_shift(productivity):
    if productivity < low_cutoff:
        return "bad"
    if productivity < high_cutoff:
        return "average"
    return "good"


df["shift_quality"] = df["productivity"].apply(label_shift)

print("\nShift quality counts:")
print(df["shift_quality"].value_counts())


# -------------------------------------------------------------------
# 2. Build the prediction model
# -------------------------------------------------------------------

# Production and truck available hours directly define the productivity
# KPI, so I leave them out of the model to avoid target leakage.
X = df.drop(columns=[
    "productivity",
    "shift_quality",
    "Production/bcm",
    "Truck available hours",
])

y = df["shift_quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nModel accuracy:", round(accuracy_score(y_test, predictions), 3))
print("\nClassification report:")
print(classification_report(y_test, predictions))


# -------------------------------------------------------------------
# 3. Simple exploratory plots
# -------------------------------------------------------------------

plt.scatter(df["total_queue_time"], df["productivity"])
plt.xlabel("Total Queue Time (seconds)")
plt.ylabel("Productivity")
plt.title("Queue Time vs Productivity")
plt.tight_layout()
plt.show()

plt.scatter(df["off_circuit_time"], df["productivity"])
plt.xlabel("Off-Circuit Travel Time (seconds)")
plt.ylabel("Productivity")
plt.title("Off-Circuit Time vs Productivity")
plt.tight_layout()
plt.show()
