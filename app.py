from utils.domain_generator import generate_domains
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
def encode_features(domain: str, year: int, tld: str) -> np.ndarray:
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
# Cached domain generation
# =============================
@st.cache_data(show_spinner=False)
def get_candidate_domains(count=200, tld="ai"):
    return generate_domains(count=count, tld=tld)

# =============================
# Cached scoring pipeline
# =============================
@st.cache_data(show_spinner=True)
def score_domains(domains, year):
    rows = []

    for d in domains:
        X = encode_features(d, year, "ai").reshape(1, -1)
        probs = rf_model.predict_proba(X)[0]

        pred_class = int(np.argmax(probs))
        ml_conf = float(np.max(probs))

        keyword = d.split(".")[0]
        news_text, news_score = fetch_news(keyword)

        # Final score (NO circular logic)
        final_score = round(
            0.6 * ml_conf +
            0.4 * news_score,
            3
        )

        # Mature pricing
        if pred_class == 0:
            base, span = 5000, 10000
        elif pred_class == 1:
            base, span = 20000, 25000
        else:
            base, span = 40000, 40000

        est_price = int(base + final_score * span)
        low = int(est_price * 0.85)
        high = int(est_price * 1.15)

        rows.append({
            "Domain": d,
            "Value Class": ["Low", "Medium", "High"][pred_class],
            "ML Score": round(ml_conf, 2),
            "News Score": round(news_score, 2),
            "Final Score": final_score,
            "Estimated Price": f"${low:,.0f} – ${high:,.0f}",
            "Latest News": news_text if news_text else infer_trend(d)
        })

    return pd.DataFrame(rows)

# =============================
# UI
# =============================
st.title("🚀 Domain Value Forecasting System")
st.caption(
    "AI-assisted system combining historical domain valuation patterns "
    "with real-time market signals."
)

domain_input = st.text_input("Enter domain name", "futuretech.ai")
year_input = st.number_input("Expected sale year", 2025, 2035, 2026)
tld_input = st.selectbox("Select TLD", sorted(tld_columns))

# SINGLE BUTTON (IMPORTANT)
predict_clicked = st.button("Predict Value", key="predict_main")

if predict_clicked:
    with st.spinner("Analyzing market signals and ranking domains..."):

        # ---- Single domain prediction ----
        X = encode_features(domain_input, year_input, tld_input).reshape(1, -1)
        probs = rf_model.predict_proba(X)[0]
        pred_class = int(np.argmax(probs))

        st.subheader("📊 Prediction Result")
        st.write(f"**Domain:** {domain_input}")
        st.write(f"**Predicted Value:** {['Low','Medium','High'][pred_class]}")
        st.write(f"**ML Confidence:** {np.max(probs):.2f}")

        # ---- Discovery & ranking ----
        candidate_domains = get_candidate_domains(count=200, tld="ai")

        if not candidate_domains:
            st.error("Domain generator returned zero domains.")
            st.stop()

        df_all = score_domains(candidate_domains, year_input)

        if df_all.empty:
            st.error("Scoring pipeline returned empty results.")
            st.stop()

        df_top5 = (
            df_all
            .sort_values("Final Score", ascending=False)
            .head(5)
            .reset_index(drop=True)
        )

        df_top5.insert(0, "Rank", range(1, len(df_top5) + 1))

    st.success(f"Scanned {len(candidate_domains)} candidate domains")

    st.subheader("🏆 Top 5 High-Potential Domain Opportunities")
    st.dataframe(df_top5, use_container_width=True)
