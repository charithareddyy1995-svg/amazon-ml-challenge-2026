import re
import unicodedata
from typing import List


LEGAL_SUFFIXES = {
    "inc",
    "incorporated",
    "llc",
    "ltd",
    "limited",
    "llp",
    "corp",
    "corporation",
    "company",
    "co",
    "private",
    "pvt",
}


def normalize_unicode(text: str) -> str:
    """Normalize Unicode while preserving useful characters."""
    if not text:
        return ""

    text = str(text)

    text = unicodedata.normalize("NFKD", text)

    text = "".join(
        char
        for char in text
        if not unicodedata.combining(char)
    )

    return text


def normalize_text(text: str) -> str:
    """
    General text normalization.

    Converts text to lowercase, removes punctuation,
    and normalizes whitespace.
    """
    if not text:
        return ""

    text = normalize_unicode(text)

    text = text.lower()

    # Replace punctuation/symbols with spaces.
    text = re.sub(r"[^a-z0-9]+", " ", text)
    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_name(text: str) -> str:
    """
    Normalize a business name.
    """
    text = normalize_text(text)

    if not text:
        return ""

    tokens = text.split()

    # Remove common legal suffixes only when they occur
    # as complete tokens.
    tokens = [
        token
        for token in tokens
        if token not in LEGAL_SUFFIXES
    ]

    return " ".join(tokens)


def normalize_address(text: str) -> str:
    """
    Normalize a business address.
    """
    text = normalize_text(text)

    if not text:
        return ""

    # Common address abbreviations.
    replacements = {
        "street": "st",
        "avenue": "ave",
        "road": "rd",
        "drive": "dr",
        "boulevard": "blvd",
        "lane": "ln",
        "highway": "hwy",
        "parkway": "pkwy",
        "place": "pl",
        "court": "ct",
        "circle": "cir",
        "terrace": "ter",
    }

    tokens = text.split()

    tokens = [
        replacements.get(token, token)
        for token in tokens
    ]

    return " ".join(tokens)


def get_name_tokens(text: str) -> List[str]:
    """Return normalized business-name tokens."""
    normalized = normalize_name(text)

    if not normalized:
        return []

    return normalized.split()


def get_address_tokens(text: str) -> List[str]:
    """Return normalized address tokens."""
    normalized = normalize_address(text)

    if not normalized:
        return []

    return normalized.split()


def character_ngrams(text: str, n: int = 3) -> List[str]:
    """
    Generate character n-grams for fuzzy/blocking features.
    """
    text = normalize_text(text)

    if not text:
        return []

    text = text.replace(" ", "")

    if len(text) < n:
        return [text]

    return [
        text[i:i + n]
        for i in range(len(text) - n + 1)
    ]


if __name__ == "__main__":
    examples = [
        "Payne Énterprises",
        "Lumay Boral Inc.",
        "Hendricks and Flowers LLC",
        "Raj Investments LLP",
    ]

    print("NAME NORMALIZATION")
    print("=" * 50)

    for value in examples:
        print(
            f"{value}  ->  {normalize_name(value)}"
        )

    print("\nADDRESS NORMALIZATION")
    print("=" * 50)

    address_examples = [
        "3315 Fremont Street, Peoria, IL",
        "630 45th Terrace, Kansas City, MO",
        "6(29), C.I.T. Colony, 2nd Main Road, Mylapore",
    ]

    for value in address_examples:
        print(
            f"{value}  ->  {normalize_address(value)}"
        )