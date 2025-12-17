import streamlit as st
import pandas as pd
import numpy as np
import joblib

from utils.features import domain_embedding

# -----------------------------
# Load saved artifacts
# -----------------------------
rf_model = joblib.load("model/rf_model.pkl")
tld_columns = joblib.load("model/tld_columns.pkl")
year_min, year_max = joblib.load("model/year_range.pkl")

# -----------------------------
# Helper functions
# -----------------------------
def encode_features(domain, year, tld):
    embed = domain_embedding(domain)

    tld_vec = np.zeros(len(tld_columns))
    if tld in tld_columns:
        tld_vec[tld_columns.index(tld)] = 1

    if year_max > year_min:
        year_norm = (year - year_min) / (year_max - year_min)
    else:
        year_norm = 0

    return np.hstack([embed, tld_vec, year_norm])

def label_to_text(label):
    return {0: "Low", 1: "Medium", 2: "High"}[label]

# -----------------------------
# Trend / News Heuristic
# -----------------------------
TREND_KEYWORDS = {
    "ai": "AI & Automation Boom",
    "health": "Digital Health Growth",
    "finance": "FinTech Expansion",
    "crypto": "Blockchain Adoption",
    "energy": "Green Energy Push",
    "robot": "Robotics & Industry 4.0",
    "cloud": "Cloud & SaaS Growth"
}

def infer_trend(domain):
    for key, trend in TREND_KEYWORDS.items():
        if key in domain.lower():
            return trend
    return "General Startup Activity"


# -----------------------------
# UI
# -----------------------------
st.title("🚀 Domain Value Forecasting System")

st.write("Predict high-potential domain names for resale")

domain_input = st.text_input("Enter domain name", "futuretech.ai")
year_input = st.number_input("Expected sale year", 2025, 2035, 2026)
tld_input = st.selectbox("Select TLD", sorted(tld_columns))

if st.button("Predict Value"):
    X = encode_features(domain_input, year_input, tld_input).reshape(1, -1)

    probs = rf_model.predict_proba(X)[0]
    pred_class = np.argmax(probs)

    st.subheader("Prediction Result")
    st.write(f"**Domain:** {domain_input}")
    st.write(f"**Predicted Value Class:** {label_to_text(pred_class)}")
    st.write(f"**Confidence:** {probs[pred_class]:.2f}")



candidate_domains = [
    "agentstack.ai",
    "quantumvault.ai",
    "greeninfra.ai",
    "neurofinance.ai",
    "healthmesh.ai",
    "roboticscloud.ai",
    "aifinancehub.ai",
    "energyai.ai",
    "datasecurity.ai",
    "climatechain.ai"
]


rows = []

rows = []

for d in candidate_domains:
    # Encode features
    X = encode_features(d, year_input, "ai").reshape(1, -1)

    # Predict probabilities
    probs = rf_model.predict_proba(X)[0]
    cls = np.argmax(probs)
    high_prob = probs[cls]   # ← CONFIDENCE

    rows.append({
        "Rank": 0,  # placeholder (fixed later)
        "Domain": d,
        "Why it matters": infer_trend(d),
        "Confidence Score": round(float(high_prob), 2),
        "Estimated Price (USD)": int(40000 + high_prob * 70000)
    })

# =========================
# AFTER LOOP (IMPORTANT)
# =========================

df = pd.DataFrame(rows)

# Sort by confidence (highest first)
df = df.sort_values(by="Confidence Score", ascending=False)

# Keep only top 5 domains
df = df.head(5).reset_index(drop=True)

# Assign proper ranking
df["Rank"] = df.index + 1

# Format price nicely
df["Estimated Price (USD)"] = df["Estimated Price (USD)"].apply(
    lambda x: f"${x:,.0f}"
)

# Display table
st.subheader("🔥 Top 5 Upcoming Domain Opportunities")
st.dataframe(
    df,
    use_container_width=True
)



df = pd.DataFrame(rows)
st.dataframe(df)



    




