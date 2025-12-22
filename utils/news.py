import os
import requests
from functools import lru_cache

API_KEY = os.getenv("NEWS_API_KEY")

@lru_cache(maxsize=256)
def fetch_news(keyword: str):
    """
    Fetch recent news headlines and compute a normalized news score.
    Returns: (headline_summary, news_score)
    """
    if not API_KEY:
        return "News API key not configured", 0.0

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": keyword,
        "language": "en",
        "pageSize": 10,
        "sortBy": "publishedAt",
        "apiKey": API_KEY,
    }

    try:
        r = requests.get(url, params=params, timeout=5)
        data = r.json()

        articles = data.get("articles", [])
        if not articles:
            return "No significant recent news coverage", 0.0

        # Take top 3 headlines
        headlines = [a["title"] for a in articles[:3]]
        headline_text = " | ".join(headlines)

        # News score based on volume (bounded)
        news_score = min(len(articles) / 10, 1.0)

        return headline_text, round(news_score, 2)

    except Exception:
        return "News fetch error", 0.0
