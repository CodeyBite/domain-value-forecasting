import os
import requests
from functools import lru_cache

API_KEY = os.getenv("NEWS_API_KEY")

@lru_cache(maxsize=128)
def fetch_news(keyword: str):
    # Fallback if API key is missing
    if not API_KEY:
        return f"Growing interest in {keyword} technologies", 0.3

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": keyword,
        "language": "en",
        "pageSize": 5,
        "sortBy": "publishedAt",
        "apiKey": API_KEY,
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        data = response.json()

        articles = data.get("articles", [])
        if not articles:
            return "No major recent news", 0.3

        headlines = [a["title"] for a in articles[:3]]
        news_text = "; ".join(headlines)

        # Simple + explainable scoring
        news_score = min(len(articles) / 5, 1.0)

        return news_text, round(news_score, 2)

    except Exception:
        return "News fetch error", 0.3
