import random
import pandas as pd

# -----------------------------
# Configuration
# -----------------------------
tlds = ["com", "ai", "io", "tech", "net", "org"]
keywords = [
    "ai", "cloud", "data", "block", "crypto", "health",
    "fin", "smart", "future", "neural", "quant", "web"
]

years = list(range(2015, 2026))

rows = []

# -----------------------------
# Generation rules
# -----------------------------
def price_rule(keyword, tld):
    base = random.randint(5000, 15000)

    if keyword in ["ai", "crypto", "block", "neural", "quant"]:
        base += random.randint(30000, 80000)

    if tld in ["ai", "io", "tech"]:
        base += random.randint(15000, 50000)

    return base

# -----------------------------
# Generate data
# -----------------------------
for _ in range(1000):
    kw = random.choice(keywords)
    name = f"{kw}{random.randint(1,999)}"
    tld = random.choice(tlds)
    domain = f"{name}.{tld}"

    price = price_rule(kw, tld)
    year = random.choice(years)

    rows.append([domain, price, year, tld])

# -----------------------------
# Save CSV
# -----------------------------
df = pd.DataFrame(rows, columns=["domain", "price_usd", "sale_year", "tld"])
df.to_csv("data/domain_sales.csv", index=False)

print("✅ Dataset generated:", df.shape)
