import streamlit as st
import pandas as pd
import numpy as np
import joblib

from utils.features import domain_embedding
from utils.news import fetch_news

# =============================
# Load trained artifacts
# =============================
rf_model = joblib.load("model/rf_model.pkl")
tld_columns = joblib.load("model/tld_columns.pkl")
year_min, year_max = joblib.load("model/year_range.pkl")

# =============================
# Trend keywords (fallback)
# =============================
TREND_KEYWORDS = {
    "ai": "AI & Automation Boom",
    "health": "Digital Health Growth",
    "finance": "FinTech Expansion",
    "quantum": "Quantum Computing Investments Rising",
    "energy": "Green Energy Push",
    "robot": "Robotics & Industry 4.0",
    "cloud": "Cloud & SaaS Growth",
    "data": "Enterprise Data & Security Demand"
}

def infer_trend(domain: str) -> str:
    d = domain.lower()
    for k, v in TREND_KEYWORDS.items():
        if k in d:
            return v
    return "General Startup & Technology Growth"

# =============================
# Feature encoding
# =============================
def encode_features(domain, year, tld):
    embed = domain_embedding(domain)

    tld_vec = np.zeros(len(tld_columns))
    if tld in tld_columns:
        tld_vec[tld_columns.index(tld)] = 1

    year_norm = (
        (year - year_min) / (year_max - year_min)
        if year_max > year_min else 0
    )

    return np.hstack([embed, tld_vec, year_norm])

# =============================
# UI
# =============================
st.title("🚀 Domain Value Forecasting System")
st.write("Predict high-potential domain names for resale using ML + trend signals")

domain_input = st.text_input("Enter domain name", "futuretech.ai")
year_input = st.number_input("Expected sale year", 2025, 2035, 2026)
tld_input = st.selectbox("Select TLD", sorted(tld_columns))

if st.button("Predict Value"):
    X = encode_features(domain_input, year_input, tld_input).reshape(1, -1)
    probs = rf_model.predict_proba(X)[0]
    pred_class = int(np.argmax(probs))

    st.subheader("Prediction Result")
    st.write(f"**Domain:** {domain_input}")
    st.write(f"**Predicted Value:** {['Low','Medium','High'][pred_class]}")
    st.write(f"**ML Confidence:** {np.max(probs):.2f}")

# =============================
# Candidate domains (research pool)
# =============================
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

for d in candidate_domains:
    X = encode_features(d, year_input, "ai").reshape(1, -1)
    probs = rf_model.predict_proba(X)[0]

    pred_class = int(np.argmax(probs))
    ml_conf = float(np.max(probs))

    keyword = d.split(".")[0]
    news_text, news_score = fetch_news(keyword)

    # Final score (balanced, conservative)
    final_score = round(
        0.6 * ml_conf +
        0.25 * news_score +
        0.15 * (0.2 if pred_class == 2 else 0.1 if pred_class == 1 else 0.0),
        3
    )

    # Price estimation (REALISTIC)
    if pred_class == 0:
        base = 15000
        span = 12000
    elif pred_class == 1:
        base = 35000
        span = 25000
    else:
        base = 60000
        span = 40000

    est_price = int(base + final_score * span)
    low = int(est_price * 0.85)
    high = int(est_price * 1.15)

    rows.append({
        "Domain": d,
        "Value": ["Low", "Medium", "High"][pred_class],
        "ML Score": round(ml_conf, 2),
        "News Score": round(news_score, 2),
        "Final Score": final_score,
        "Estimated Price": f"${low:,.0f} – ${high:,.0f}",
        "Reason": news_text if news_text else infer_trend(d)
    })

# =============================
# Final ranking table
# =============================
df = pd.DataFrame(rows)

df = (
    df.sort_values("Final Score", ascending=False)
      .head(5)
      .reset_index(drop=True)
)

df.insert(0, "Rank", df.index + 1)

st.subheader("🏆 Top 5 High-Potential Domain Opportunities")
st.dataframe(df, use_container_width=True)
