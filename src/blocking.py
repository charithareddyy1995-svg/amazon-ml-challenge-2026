import pandas as pd
from pathlib import Path

from preprocessing import normalize_name, normalize_address


TRAIN_DIR = Path("dataset/train")

SOURCE1 = TRAIN_DIR / "train_source1.tsv"
SOURCE2 = TRAIN_DIR / "train_source2.tsv"
SOURCE3 = TRAIN_DIR / "train_source3.tsv"


def name_block_key(name):
    """
    Create a simple blocking key from the first
    few normalized name tokens.
    """
    name = normalize_name(name)

    if not name:
        return ""

    tokens = name.split()

    # Use up to the first two meaningful tokens.
    return " ".join(tokens[:2])


def address_block_key(address):
    """
    Create a simple blocking key from address tokens.
    """
    address = normalize_address(address)

    if not address:
        return ""

    tokens = address.split()

    # Keep the first few tokens.
    return " ".join(tokens[:3])


def create_block_keys(df):
    """
    Add blocking columns to a dataframe.
    """

    df = df.copy()

    df["name_block"] = (
        df["business_name"]
        .fillna("")
        .map(name_block_key)
    )

    df["address_block"] = (
        df["business_address"]
        .fillna("")
        .map(address_block_key)
    )

    df["country_block"] = (
        df["country"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    return df


def build_block_index(df):
    """
    Build inverted indexes for candidate generation.

    Returns:
        name_index
        address_index
    """

    name_index = {}
    address_index = {}

    for _, row in df.iterrows():

        entity_id = row["entity_id"]

        country = row["country_block"]

        name_key = row["name_block"]

        address_key = row["address_block"]

        # Country + name block
        if name_key:
            key = (country, name_key)

            name_index.setdefault(key, []).append(
                entity_id
            )

        # Country + address block
        if address_key:
            key = (country, address_key)

            address_index.setdefault(key, []).append(
                entity_id
            )

    return name_index, address_index


def generate_candidates(
    source1_row,
    name_index,
    address_index
):
    """
    Generate candidate IDs for one S1 record.

    Candidates from multiple blocking strategies
    are combined using a set.
    """

    candidates = set()

    country = str(
        source1_row.get("country", "")
    ).lower().strip()

    name_key = name_block_key(
        source1_row.get("business_name", "")
    )

    address_key = address_block_key(
        source1_row.get("business_address", "")
    )

    # Blocking strategy 1:
    # country + normalized name prefix
    if name_key:
        key = (country, name_key)

        candidates.update(
            name_index.get(key, [])
        )

    # Blocking strategy 2:
    # country + normalized address prefix
    if address_key:
        key = (country, address_key)

        candidates.update(
            address_index.get(key, [])
        )

    return candidates


if __name__ == "__main__":

    print("=" * 70)
    print("BLOCKING TEST")
    print("=" * 70)

    # Small sample only for testing.
    print("\nReading sample Source 2 data...")

    source2 = pd.read_csv(
        SOURCE2,
        sep="\t",
        dtype=str,
        nrows=10000
    )

    source2 = create_block_keys(source2)

    name_index, address_index = build_block_index(
        source2
    )

    print(
        f"Source 2 records: {len(source2):,}"
    )

    print(
        f"Name blocks: {len(name_index):,}"
    )

    print(
        f"Address blocks: {len(address_index):,}"
    )

    # Test using first S1 record.
    print("\nReading one Source 1 record...")

    source1 = pd.read_csv(
        SOURCE1,
        sep="\t",
        dtype=str,
        nrows=1
    )

    row = source1.iloc[0].to_dict()

    candidates = generate_candidates(
        row,
        name_index,
        address_index
    )

    print(
        f"\nS1 ID: {row['entity_id']}"
    )

    print(
        f"Business name: {row['business_name']}"
    )

    print(
        f"Candidates found in 10,000-record sample: "
        f"{len(candidates)}"
    )

    print("\nBlocking test complete.")