import pandas as pd
import numpy as np

from utils.features import domain_embedding

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("data/domain_sales.csv")

# -----------------------------
# Enforce correct schema
# -----------------------------
df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")
df["sale_year"] = pd.to_numeric(df["sale_year"], errors="coerce")

df = df.dropna()

# Keep required columns
df = df[["domain", "price_usd", "sale_year", "tld"]]

print("Dataset shape:", df.shape)
print(df.dtypes)

# -----------------------------
# Create multi-class labels
# -----------------------------
q1 = df["price_usd"].quantile(0.33)
q2 = df["price_usd"].quantile(0.66)

def price_bucket(p):
    if p <= q1:
        return 0   # Low
    elif p <= q2:
        return 1   # Medium
    else:
        return 2   # High

df["label"] = df["price_usd"].apply(price_bucket)

print("\nPrice thresholds:")
print(f"Low ≤ {q1:.0f}, Medium ≤ {q2:.0f}, High > {q2:.0f}")
print("\nClass distribution:")
print(df["label"].value_counts())

# -----------------------------
# Feature engineering
# -----------------------------
embeddings = np.array([domain_embedding(d) for d in df["domain"]])

tld_encoded = pd.get_dummies(df["tld"], drop_first=True)

year_min = df["sale_year"].min()
year_max = df["sale_year"].max()

if year_max > year_min:
    year_norm = (df["sale_year"] - year_min) / (year_max - year_min)
else:
    year_norm = np.zeros(len(df))

X = np.hstack([
    embeddings,
    tld_encoded.values,
    year_norm.values.reshape(-1, 1)
])

y = df["label"].values

# -----------------------------
# Train / test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Train model
# -----------------------------
model = LogisticRegression(
    max_iter=2000,
    solver="lbfgs"
)

model.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = model.predict(X_test)

print("\nBaseline Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
