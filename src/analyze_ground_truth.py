import pandas as pd
from pathlib import Path
from collections import Counter

GROUND_TRUTH = Path("dataset/train/train_ground_truth.tsv")

match_counts = Counter()
total_rows = 0
empty_matches = 0
s2_matches = 0
s3_matches = 0

for chunk in pd.read_csv(
    GROUND_TRUTH,
    sep="\t",
    chunksize=200_000,
    dtype=str
):
    for value in chunk["matched_entity_ids"].fillna(""):
        value = value.strip()

        if not value:
            empty_matches += 1
            match_counts[0] += 1
            continue

        ids = [x.strip() for x in value.split(",") if x.strip()]

        match_counts[len(ids)] += 1

        for entity_id in ids:
            if entity_id.startswith("S2-"):
                s2_matches += 1
            elif entity_id.startswith("S3-"):
                s3_matches += 1

    total_rows += len(chunk)

print("=" * 60)
print("GROUND TRUTH ANALYSIS")
print("=" * 60)

print(f"Total S1 entities: {total_rows:,}")
print(f"No-match S1 entities: {empty_matches:,}")
print(f"S2 matches: {s2_matches:,}")
print(f"S3 matches: {s3_matches:,}")

print("\nNumber of matches per S1:")
for count in sorted(match_counts):
    print(f"{count} matches -> {match_counts[count]:,} S1 entities")

print("\nDone.")