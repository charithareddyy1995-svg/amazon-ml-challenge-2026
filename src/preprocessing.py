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
    """Normalize Unicode while preserving non-Latin scripts and useful letters."""
    if text is None:
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKD", text)

    result = []
    for char in text:
        if unicodedata.combining(char):
            # Drop accents from Latin letters, but keep combining marks for scripts
            # such as Hindi and Tamil so meaningful vowel signs remain.
            if result and result[-1].isascii() and result[-1].isalpha():
                continue
        result.append(char)

    return "".join(result)


def normalize_text(text: str) -> str:
    """
    General text normalization.

    Preserves letters and digits from all scripts while removing punctuation
    and symbols without using a hardcoded Latin-only regex.
    """
    if text is None:
        return ""

    text = normalize_unicode(text)
    text = text.lower()

    normalized_chars = []
    for ch in text:
        if ch.isalnum() or unicodedata.category(ch).startswith("M"):
            normalized_chars.append(ch)
        elif ch.isspace():
            normalized_chars.append(" ")
        else:
            normalized_chars.append(" ")

    text = "".join(normalized_chars)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_name(text: str) -> str:
    """Normalize a business name while keeping non-Latin scripts."""
    text = normalize_text(text)

    if not text:
        return ""

    tokens = text.split()
    tokens = [token for token in tokens if token not in LEGAL_SUFFIXES]
    return " ".join(tokens)


def normalize_address(text: str) -> str:
    """Normalize a business address while keeping Unicode scripts intact."""
    text = normalize_text(text)

    if not text:
        return ""

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
    tokens = [replacements.get(token, token) for token in tokens]
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
    """Generate character n-grams for string matching while keeping Unicode letters."""
    text = normalize_text(text)

    if not text:
        return []

    text = text.replace(" ", "")

    if len(text) < n:
        return [text]

    return [text[i:i + n] for i in range(len(text) - n + 1)]


if __name__ == "__main__":
    examples = [
        "Payne Énterprises",
        "होटल Enterprises लिमिटेड",
        "தமிழ் Enterprises",
        "Raj Investments LLP",
        "",
        None,
    ]

    print("NAME NORMALIZATION")
    print("=" * 60)
    for value in examples:
        print(f"{value!r}  ->  {normalize_name(value)}")

    print("\nADDRESS NORMALIZATION")
    print("=" * 60)
    address_examples = [
        "3315 Fremont Street, Peoria, IL",
        "630 45th Terrace, Kansas City, MO",
        "6(29), C.I.T. Colony, 2nd Main Road, Mylapore",
        "சிங்காரி வீதி, சென்னை",
        "",
        None,
    ]

    for value in address_examples:
        print(f"{value!r}  ->  {normalize_address(value)}")