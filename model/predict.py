import sys
import numpy as np
import pandas as pd
import joblib

from utils.features import domain_embedding

# -----------------------------
# Load trained model & metadata
# -----------------------------
model = joblib.load("model/domain_value_model.pkl")
tld_columns = joblib.load("model/tld_columns.pkl")
year_min, year_max = joblib.load("model/year_range.pkl")

# -----------------------------
# Read input
# -----------------------------
if len(sys.argv) != 4:
    print("Usage: python model/predict.py <domain> <sale_year> <tld>")
    sys.exit(1)

domain = sys.argv[1]
sale_year = int(sys.argv[2])
tld = sys.argv[3]

# -----------------------------
# Feature engineering
# -----------------------------
embed = np.array(domain_embedding(domain)).reshape(1, -1)

tld_df = pd.DataFrame([[tld]], columns=["tld"])
tld_encoded = pd.get_dummies(tld_df)
tld_encoded = tld_encoded.reindex(columns=tld_columns, fill_value=0)

if year_max > year_min:
    year_norm = (sale_year - year_min) / (year_max - year_min)
else:
    year_norm = 0

X = np.hstack([
    embed,
    tld_encoded.values,
    np.array([[year_norm]])
])

# -----------------------------
# Predict
# -----------------------------
pred = model.predict(X)[0]
prob = model.predict_proba(X)[0][pred]

labels = {0: "Low", 1: "Medium", 2: "High"}

print("\nPrediction Result")
print("-----------------")
print(f"Domain: {domain}")
print(f"Predicted value class: {labels[pred]}")
print(f"Confidence: {prob:.2f}")
