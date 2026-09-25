import pandas as pd
import re
import unicodedata
from pathlib import Path


TRAIN_DIR = Path("dataset/train")

SOURCE1 = TRAIN_DIR / "train_source1.tsv"
SOURCE2 = TRAIN_DIR / "train_source2.tsv"
SOURCE3 = TRAIN_DIR / "train_source3.tsv"
GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"


def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).lower()

    # Remove accents
    value = unicodedata.normalize("NFKD", value)
    value = "".join(
        c for c in value
        if not unicodedata.combining(c)
    )

    # Replace punctuation with spaces
    value = re.sub(r"[^a-z0-9]+", " ", value)

    # Remove extra spaces
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


print("=" * 70)
print("NORMALIZATION PATTERN ANALYSIS")
print("=" * 70)

# ---------------------------------------------------------
# Read a manageable ground-truth sample
# ---------------------------------------------------------

print("\nReading ground truth...")

gt = pd.read_csv(
    GROUND_TRUTH,
    sep="\t",
    dtype=str,
    nrows=5000
)

gt = gt[
    gt["matched_entity_ids"].fillna("").str.strip() != ""
]

# Take first 100 S1 entities with matches
gt = gt.head(100)

s1_ids = set(gt["source1_entity_id"])
s2_ids = set()
s3_ids = set()

for value in gt["matched_entity_ids"]:
    for entity_id in value.split(","):
        entity_id = entity_id.strip()

        if entity_id.startswith("S2-"):
            s2_ids.add(entity_id)

        elif entity_id.startswith("S3-"):
            s3_ids.add(entity_id)


# ---------------------------------------------------------
# Find records
# ---------------------------------------------------------

def find_records(file_path, ids):
    records = []

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        chunksize=200_000
    ):
        found = chunk[
            chunk["entity_id"].isin(ids)
        ]

        if not found.empty:
            records.append(found)

    if records:
        return pd.concat(records, ignore_index=True)

    return pd.DataFrame(
        columns=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    )


print("Searching Source 1...")
s1_df = find_records(SOURCE1, s1_ids)

print("Searching Source 2...")
s2_df = find_records(SOURCE2, s2_ids)

print("Searching Source 3...")
s3_df = find_records(SOURCE3, s3_ids)


s1_lookup = s1_df.set_index("entity_id").to_dict("index")
s2_lookup = s2_df.set_index("entity_id").to_dict("index")
s3_lookup = s3_df.set_index("entity_id").to_dict("index")


# ---------------------------------------------------------
# Calculate statistics
# ---------------------------------------------------------

total_pairs = 0

country_same = 0
name_exact = 0
address_exact = 0

name_jaccard_sum = 0
address_jaccard_sum = 0

name_high = 0
address_high = 0

missing_name = 0
missing_address = 0


for _, row in gt.iterrows():

    s1 = s1_lookup.get(row["source1_entity_id"])

    if not s1:
        continue

    for match_id in row["matched_entity_ids"].split(","):

        match_id = match_id.strip()

        if match_id.startswith("S2-"):
            match = s2_lookup.get(match_id)
        elif match_id.startswith("S3-"):
            match = s3_lookup.get(match_id)
        else:
            continue

        if not match:
            continue

        total_pairs += 1

        s1_name = s1.get("business_name", "")
        s2_name = match.get("business_name", "")

        s1_address = s1.get("business_address", "")
        s2_address = match.get("business_address", "")

        s1_country = str(s1.get("country", "")).strip().lower()
        s2_country = str(match.get("country", "")).strip().lower()

        # Country
        if s1_country and s2_country:
            if s1_country == s2_country:
                country_same += 1

        # Missing fields
        if not normalize_text(s1_name) or not normalize_text(s2_name):
            missing_name += 1

        if not normalize_text(s1_address) or not normalize_text(s2_address):
            missing_address += 1

        # Exact normalized name
        n1 = normalize_text(s1_name)
        n2 = normalize_text(s2_name)

        if n1 and n2 and n1 == n2:
            name_exact += 1

        # Exact normalized address
        a1 = normalize_text(s1_address)
        a2 = normalize_text(s2_address)

        if a1 and a2 and a1 == a2:
            address_exact += 1

        # Jaccard
        name_score = jaccard(s1_name, s2_name)
        address_score = jaccard(s1_address, s2_address)

        name_jaccard_sum += name_score
        address_jaccard_sum += address_score

        if name_score >= 0.5:
            name_high += 1

        if address_score >= 0.5:
            address_high += 1


# ---------------------------------------------------------
# Print results
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(f"\nPositive pairs analyzed: {total_pairs:,}")

if total_pairs > 0:

    print(
        f"\nCountry exactly same: "
        f"{country_same:,} "
        f"({country_same / total_pairs:.2%})"
    )

    print(
        f"Normalized name exactly same: "
        f"{name_exact:,} "
        f"({name_exact / total_pairs:.2%})"
    )

    print(
        f"Normalized address exactly same: "
        f"{address_exact:,} "
        f"({address_exact / total_pairs:.2%})"
    )

    print(
        f"\nName Jaccard >= 0.50: "
        f"{name_high:,} "
        f"({name_high / total_pairs:.2%})"
    )

    print(
        f"Address Jaccard >= 0.50: "
        f"{address_high:,} "
        f"({address_high / total_pairs:.2%})"
    )

    print(
        f"\nAverage name Jaccard: "
        f"{name_jaccard_sum / total_pairs:.4f}"
    )

    print(
        f"Average address Jaccard: "
        f"{address_jaccard_sum / total_pairs:.4f}"
    )

    print(
        f"\nPairs with missing name: "
        f"{missing_name:,} "
        f"({missing_name / total_pairs:.2%})"
    )

    print(
        f"Pairs with missing address: "
        f"{missing_address:,} "
        f"({missing_address / total_pairs:.2%})"
    )

print("\nDone.")