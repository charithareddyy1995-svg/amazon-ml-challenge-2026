import pandas as pd
from pathlib import Path

DATASET = Path("dataset")


def inspect_file(path):
    print("\n" + "=" * 70)
    print(f"FILE: {path}")
    print("=" * 70)

    # Read only the first 5 rows
    df = pd.read_csv(path, sep="\t", nrows=5)

    print("Columns:")
    print(list(df.columns))

    print("\nFirst 5 rows:")
    print(df.to_string(index=False))

    # Count rows without loading the entire file
    with open(path, "r", encoding="utf-8") as f:
        row_count = sum(1 for _ in f) - 1

    print(f"\nTotal rows: {row_count:,}")


files = [
    DATASET / "train" / "train_source1.tsv",
    DATASET / "train" / "train_source2.tsv",
    DATASET / "train" / "train_source3.tsv",
    DATASET / "train" / "train_ground_truth.tsv",
    DATASET / "test" / "test_source1.tsv",
    DATASET / "test" / "test_source2.tsv",
    DATASET / "test" / "test_source3.tsv",
]


for file in files:
    inspect_file(file)