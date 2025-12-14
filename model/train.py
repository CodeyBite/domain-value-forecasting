import pandas as pd
import numpy as np
from utils.features import domain_embedding
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Load historical domain sales
# -----------------------------
df = pd.read_csv("data/domain_sales.csv")

# Keep only required columns
df = df[["domain", "price"]]

# Remove missing values
df = df.dropna()

# -----------------------------
# Create binary labels
# -----------------------------
median_price = df["price"].median()

df["label"] = (df["price"] >= median_price).astype(int)

print(f"Median price used for labeling: {median_price}")
print(df["label"].value_counts())

# -----------------------------
# Convert domains to embeddings
# -----------------------------
X = np.array([domain_embedding(d) for d in df["domain"]])
y = df["label"].values

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Train baseline model
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -----------------------------
# Evaluate model
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nBaseline Model Accuracy: {accuracy:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))