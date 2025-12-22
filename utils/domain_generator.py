import random

SECTORS = {
    "ai": ["ai", "agent", "neural", "smart", "bot"],
    "health": ["health", "med", "care", "bio"],
    "finance": ["fin", "pay", "wealth", "capital"],
    "energy": ["green", "energy", "solar", "climate"],
    "data": ["data", "secure", "cloud", "vault"]
}

SUFFIXES = [
    "hub", "labs", "tech", "systems",
    "stack", "vault", "network",
    "platform", "core"
]

def generate_domains(count=200, tld="ai"):
    domains = set()
    while len(domains) < count:
        sector_words = random.choice(list(SECTORS.values()))
        w1 = random.choice(sector_words)
        w2 = random.choice(SUFFIXES)
        domains.add(f"{w1}{w2}.{tld}")
    return list(domains)
