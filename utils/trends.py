from pytrends.request import TrendReq

def trend_score(keyword):
    pytrends = TrendReq(hl="en-US", tz=330)
    pytrends.build_payload([keyword], timeframe="today 3-m")
    data = pytrends.interest_over_time()

    if data.empty:
        return 0.3

    return min(data[keyword].mean() / 100, 1.0)
