import pandas as pd
from pathlib import Path


TRAIN_DIR = Path("dataset/train")

GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"
SOURCE1 = TRAIN_DIR / "train_source1.tsv"
SOURCE2 = TRAIN_DIR / "train_source2.tsv"
SOURCE3 = TRAIN_DIR / "train_source3.tsv"


# ---------------------------------------------------------
# 1. Read a small sample of ground-truth matches
# ---------------------------------------------------------

print("Reading ground truth sample...")

gt = pd.read_csv(
    GROUND_TRUTH,
    sep="\t",
    dtype=str,
    nrows=1000
)

# Keep only S1 entities that actually have matches
gt = gt[
    gt["matched_entity_ids"].fillna("").str.strip() != ""
]

# Take first 10 matched S1 entities
gt = gt.head(10)

print(f"Selected {len(gt)} S1 entities.")


# ---------------------------------------------------------
# 2. Collect the S1, S2 and S3 IDs we need
# ---------------------------------------------------------

s1_ids = set(gt["source1_entity_id"])

s2_ids = set()
s3_ids = set()

for value in gt["matched_entity_ids"]:
    ids = [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]

    for entity_id in ids:
        if entity_id.startswith("S2-"):
            s2_ids.add(entity_id)

        elif entity_id.startswith("S3-"):
            s3_ids.add(entity_id)


print(f"S1 IDs needed: {len(s1_ids)}")
print(f"S2 IDs needed: {len(s2_ids)}")
print(f"S3 IDs needed: {len(s3_ids)}")


# ---------------------------------------------------------
# 3. Find those records in Source 1
# ---------------------------------------------------------

print("\nSearching Source 1...")

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


# ---------------------------------------------------------
# 4. Find those records in Source 2
# ---------------------------------------------------------

print("Searching Source 2...")

s2_records = []

for chunk in pd.read_csv(
    SOURCE2,
    sep="\t",
    dtype=str,
    chunksize=200_000
):
    found = chunk[
        chunk["entity_id"].isin(s2_ids)
    ]

    if not found.empty:
        s2_records.append(found)

s2_df = pd.concat(s2_records, ignore_index=True)


# ---------------------------------------------------------
# 5. Find those records in Source 3
# ---------------------------------------------------------

print("Searching Source 3...")

s3_records = []

for chunk in pd.read_csv(
    SOURCE3,
    sep="\t",
    dtype=str,
    chunksize=200_000
):
    found = chunk[
        chunk["entity_id"].isin(s3_ids)
    ]

    if not found.empty:
        s3_records.append(found)


if s3_records:
    s3_df = pd.concat(s3_records, ignore_index=True)
else:
    s3_df = pd.DataFrame(
        columns=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    )


# ---------------------------------------------------------
# 6. Create lookup dictionaries
# ---------------------------------------------------------

s1_lookup = s1_df.set_index("entity_id").to_dict("index")
s2_lookup = s2_df.set_index("entity_id").to_dict("index")
s3_lookup = s3_df.set_index("entity_id").to_dict("index")


# ---------------------------------------------------------
# 7. Print actual positive matches
# ---------------------------------------------------------

print("\n")
print("=" * 100)
print("ACTUAL POSITIVE MATCH EXAMPLES")
print("=" * 100)

for _, row in gt.iterrows():

    s1_id = row["source1_entity_id"]

    print("\n" + "-" * 100)

    s1 = s1_lookup.get(s1_id)

    print(f"S1 ID: {s1_id}")

    if s1:
        print(f"S1 Name:    {s1.get('business_name', '')}")
        print(f"S1 Address: {s1.get('business_address', '')}")
        print(f"S1 Country: {s1.get('country', '')}")

    print("\nMatched records:")

    matched_ids = [
        x.strip()
        for x in row["matched_entity_ids"].split(",")
        if x.strip()
    ]

    for match_id in matched_ids:

        if match_id.startswith("S2-"):
            record = s2_lookup.get(match_id)
            source = "S2"

        elif match_id.startswith("S3-"):
            record = s3_lookup.get(match_id)
            source = "S3"

        else:
            continue

        if record:
            print(f"\n{source} ID: {match_id}")
            print(
                f"Name:    {record.get('business_name', '')}"
            )
            print(
                f"Address: {record.get('business_address', '')}"
            )
            print(
                f"Country: {record.get('country', '')}"
            )

print("\n" + "=" * 100)
print("Done.")