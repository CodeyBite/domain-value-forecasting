import re
from sentence_transformers import SentenceTransformer

# Load the embedding model once (important for performance)
model = SentenceTransformer("all-MiniLM-L6-v2")

def clean_domain(domain: str) -> str:
    """
    Clean domain name text for embedding.
    Example: 'Smart-AI123.com' -> 'smart ai123'
    """
    domain = domain.lower()
    domain = domain.replace(".com", "")
    domain = re.sub(r"[^a-z0-9]", " ", domain)
    return domain.strip()

def domain_embedding(domain: str):
    """
    Convert a domain name into a numerical embedding vector.
    """
    cleaned = clean_domain(domain)
    return model.encode(cleaned)