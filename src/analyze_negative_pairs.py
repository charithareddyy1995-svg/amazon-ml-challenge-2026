import pandas as pd
import re
import unicodedata
import random
from pathlib import Path


TRAIN_DIR = Path("dataset/train")

SOURCE1 = TRAIN_DIR / "train_source1.tsv"
SOURCE2 = TRAIN_DIR / "train_source2.tsv"
GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"


def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).lower()

    value = unicodedata.normalize("NFKD", value)
    value = "".join(
        c for c in value
        if not unicodedata.combining(c)
    )

    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value


def token_set(value):
    return set(normalize_text(value).split())


def jaccard(a, b):
    a = token_set(a)
    b = token_set(b)

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


# ---------------------------------------------------------
# Read 100 S1 entities and their true S2 matches
# ---------------------------------------------------------

print("Reading ground truth...")

gt = pd.read_csv(
    GROUND_TRUTH,
    sep="\t",
    dtype=str,
    nrows=5000
)

gt = gt.head(100)

s1_ids = set(gt["source1_entity_id"])

true_pairs = set()

for _, row in gt.iterrows():

    s1_id = row["source1_entity_id"]

    for match_id in str(
        row["matched_entity_ids"]
    ).split(","):

        match_id = match_id.strip()

        if match_id.startswith("S2-"):
            true_pairs.add((s1_id, match_id))


# ---------------------------------------------------------
# Find S1 records
# ---------------------------------------------------------

print("Searching Source 1...")

s1_records = []

for chunk in pd.read_csv(
    SOURCE1,
    sep="\t",
    dtype=str,
    chunksize=200_000
):
    found = chunk[
        chunk["entity_id"].isin(s1_ids)
    ]

    if not found.empty:
        s1_records.append(found)

s1_df = pd.concat(s1_records, ignore_index=True)

s1_lookup = s1_df.set_index("entity_id").to_dict("index")


# ---------------------------------------------------------
# Select random S2 records as negative candidates
# ---------------------------------------------------------

print("Selecting random Source 2 records...")

s2_sample = pd.read_csv(
    SOURCE2,
    sep="\t",
    dtype=str,
    nrows=200_000
)

s2_records = s2_sample.to_dict("records")


# ---------------------------------------------------------
# Calculate negative similarities
# ---------------------------------------------------------

random.seed(42)

negative_scores = []

for s1_id in s1_ids:

    s1 = s1_lookup.get(s1_id)

    if not s1:
        continue

    # Select 10 random S2 records
    candidates = random.sample(
        s2_records,
        min(10, len(s2_records))
    )

    for s2 in candidates:

        pair = (s1_id, s2["entity_id"])

        # Never treat known positive as negative
        if pair in true_pairs:
            continue

        name_score = jaccard(
            s1["business_name"],
            s2["business_name"]
        )

        address_score = jaccard(
            s1["business_address"],
            s2["business_address"]
        )

        country_same = (
            str(s1["country"]).strip().lower()
            ==
            str(s2["country"]).strip().lower()
        )

        negative_scores.append(
            (
                name_score,
                address_score,
                country_same
            )
        )


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

df = pd.DataFrame(
    negative_scores,
    columns=[
        "name_score",
        "address_score",
        "country_same"
    ]
)

print("\n" + "=" * 70)
print("NEGATIVE PAIR ANALYSIS")
print("=" * 70)

print(f"\nNegative pairs analyzed: {len(df):,}")

print(
    f"\nAverage name Jaccard: "
    f"{df['name_score'].mean():.4f}"
)

print(
    f"Average address Jaccard: "
    f"{df['address_score'].mean():.4f}"
)

print(
    f"Name Jaccard >= 0.50: "
    f"{(df['name_score'] >= 0.50).mean():.2%}"
)

print(
    f"Address Jaccard >= 0.50: "
    f"{(df['address_score'] >= 0.50).mean():.2%}"
)

print(
    f"Country same: "
    f"{df['country_same'].mean():.2%}"
)

print("\nDone.")