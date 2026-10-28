import re


def clean(text: str) -> str:
    """Lowercase, remove symbols, collapse spaces. Keeps apostrophes and letters."""
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()
