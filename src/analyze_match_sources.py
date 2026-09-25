import pandas as pd
from pathlib import Path
from collections import Counter

GROUND_TRUTH = Path("dataset/train/train_ground_truth.tsv")

patterns = Counter()

for chunk in pd.read_csv(
    GROUND_TRUTH,
    sep="\t",
    chunksize=200_000,
    dtype=str
):
    for value in chunk["matched_entity_ids"].fillna(""):
        value = value.strip()

        if not value:
            patterns["none"] += 1
            continue

        ids = [x.strip() for x in value.split(",") if x.strip()]

        has_s2 = any(x.startswith("S2-") for x in ids)
        has_s3 = any(x.startswith("S3-") for x in ids)

        if has_s2 and has_s3:
            patterns["both S2 and S3"] += 1
        elif has_s2:
            patterns["S2 only"] += 1
        elif has_s3:
            patterns["S3 only"] += 1

print("=" * 60)
print("MATCH SOURCE ANALYSIS")
print("=" * 60)

for key, value in patterns.items():
    print(f"{key}: {value:,}")

print("\nDone.")