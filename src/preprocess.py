"""Text preprocessing pipeline for sentiment classification."""

import re
from bs4 import BeautifulSoup


def clean_review(text: str) -> str:
    """Clean a single review: remove HTML, lowercase, strip noise."""
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Remove non-alphabetic characters (keep spaces)
    text = re.sub(r"[^a-z\s]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text
