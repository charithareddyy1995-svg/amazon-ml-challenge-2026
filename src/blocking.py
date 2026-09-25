import pandas as pd
from collections import defaultdict
from pathlib import Path

from preprocessing import (
    normalize_address,
    normalize_name,
    character_ngrams,
)


TRAIN_DIR = Path("dataset/train")

SOURCE1 = TRAIN_DIR / "train_source1.tsv"
SOURCE2 = TRAIN_DIR / "train_source2.tsv"
SOURCE3 = TRAIN_DIR / "train_source3.tsv"


def normalize_country(country):
    """Normalize a country string for blocking."""
    if country is None:
        return ""
    return str(country).lower().strip()


def legal_suffix_tokens(tokens):
    """Return tokens with common legal suffixes stripped."""
    from preprocessing import LEGAL_SUFFIXES

    return [t for t in tokens if t not in LEGAL_SUFFIXES]


def name_block_key(name):
    """Create a simple two-token name block key."""
    name = normalize_name(name)
    if not name:
        return ""

    tokens = legal_suffix_tokens(name.split())
    if not tokens:
        return ""
    return " ".join(tokens[:2])


def address_block_key(address):
    """Create a three-token address block key."""
    address = normalize_address(address)
    if not address:
        return ""

    tokens = address.split()
    if not tokens:
        return ""
    return " ".join(tokens[:3])


def first_name_token(name):
    """Return the first meaningful token from a normalized name."""
    name = normalize_name(name)
    if not name:
        return ""

    tokens = legal_suffix_tokens(name.split())
    return tokens[0] if tokens else ""


def name_ngram_key(name, n=3):
    """Return a short character n-gram from the normalized name."""
    name = normalize_name(name)
    if not name:
        return ""

    grams = character_ngrams(name, n=n)
    if not grams:
        return ""
    return grams[0]


def address_numeric_key(address):
    """Return the leading numeric token from an address if available."""
    address = normalize_address(address)
    if not address:
        return ""

    tokens = [token for token in address.split() if any(ch.isdigit() for ch in token)]
    return tokens[0] if tokens else ""


def candidate_block_keys(name, address):
    """Generate multiple blocking keys for a record."""
    keys = set()

    name_key = name_block_key(name)
    if name_key:
        keys.add(name_key)

    first_token = first_name_token(name)
    if first_token:
        keys.add(first_token)

    ngram_key = name_ngram_key(name, n=3)
    if ngram_key:
        keys.add(ngram_key)

    address_key = address_block_key(address)
    if address_key:
        keys.add(address_key)

    num_key = address_numeric_key(address)
    if num_key:
        keys.add(num_key)

    return sorted(k for k in keys if k)


def create_block_keys(df):
    """Add blocking columns to a dataframe."""
    df = df.copy()
    df["country_block"] = (
        df["country"].fillna("").astype(str).str.lower().str.strip()
    )
    df["name_block"] = df["business_name"].fillna("").map(name_block_key)
    df["address_block"] = df["business_address"].fillna("").map(address_block_key)
    df["name_token_block"] = df["business_name"].fillna("").map(first_name_token)
    df["name_ngram_block"] = df["business_name"].fillna("").map(lambda x: name_ngram_key(x, n=3))
    df["address_numeric_block"] = df["business_address"].fillna("").map(address_numeric_key)
    return df


def build_block_index(df):
    """Build country + blocking-key inverted indexes using sets for efficiency."""
    name_index = defaultdict(set)
    address_index = defaultdict(set)

    for row in df.itertuples(index=False):
        entity_id = getattr(row, "entity_id")
        country = normalize_country(getattr(row, "country_block", ""))

        name_key = getattr(row, "name_block", "")
        if name_key:
            name_index[(country, name_key)].add(entity_id)

        address_key = getattr(row, "address_block", "")
        if address_key:
            address_index[(country, address_key)].add(entity_id)

        first_token = getattr(row, "name_token_block", "")
        if first_token:
            name_index[(country, first_token)].add(entity_id)

        ngram_key = getattr(row, "name_ngram_block", "")
        if ngram_key:
            name_index[(country, ngram_key)].add(entity_id)

        numeric_key = getattr(row, "address_numeric_block", "")
        if numeric_key:
            address_index[(country, numeric_key)].add(entity_id)

    return name_index, address_index


def generate_candidates(source1_row, name_index, address_index):
    """Generate candidate IDs for one S1 record using multiple blocking signals."""
    candidates = set()
    country = normalize_country(source1_row.get("country", ""))

    for key in candidate_block_keys(
        source1_row.get("business_name", ""),
        source1_row.get("business_address", ""),
    ):
        if country:
            candidates.update(name_index.get((country, key), set()))
            candidates.update(address_index.get((country, key), set()))
        else:
            candidates.update(name_index.get(("", key), set()))
            candidates.update(address_index.get(("", key), set()))

    return candidates


if __name__ == "__main__":
    print("=" * 70)
    print("BLOCKING TEST")
    print("=" * 70)

    print("\nReading sample Source 2 data...")
    source2 = pd.read_csv(SOURCE2, sep="\t", dtype=str, nrows=10000)
    source2 = create_block_keys(source2)
    name_index, address_index = build_block_index(source2)

    print(f"Source 2 records: {len(source2):,}")
    print(f"Name blocks: {len(name_index):,}")
    print(f"Address blocks: {len(address_index):,}")

    print("\nReading one Source 1 record...")
    source1 = pd.read_csv(SOURCE1, sep="\t", dtype=str, nrows=1)
    row = source1.iloc[0].to_dict()
    candidates = generate_candidates(row, name_index, address_index)

    print(f"\nS1 ID: {row['entity_id']}")
    print(f"Business name: {row['business_name']}")
    print(f"Candidates found in 10,000-record sample: {len(candidates)}")
    print("\nBlocking test complete.")