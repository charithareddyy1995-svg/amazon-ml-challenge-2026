import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from blocking import candidate_block_keys, normalize_country


TRAIN_DIR = Path("dataset/train")
GROUND_TRUTH_PATH = TRAIN_DIR / "train_ground_truth.tsv"
SOURCE1_PATH = TRAIN_DIR / "train_source1.tsv"
SOURCE2_PATH = TRAIN_DIR / "train_source2.tsv"
SOURCE3_PATH = TRAIN_DIR / "train_source3.tsv"


def parse_match_ids(value):
    """Parse comma-separated match IDs into a list of clean IDs."""
    if value is None or str(value).strip() == "":
        return []

    return [item.strip() for item in str(value).split(",") if item.strip()]


def build_source_index(source_path):
    """Build a candidate index for one source using multiple blocking keys."""
    index = defaultdict(set)

    for chunk in pd.read_csv(source_path, sep="\t", dtype=str, chunksize=200_000):
        subset = chunk[["entity_id", "business_name", "business_address", "country"]].fillna("")

        for row in subset.itertuples(index=False):
            entity_id = getattr(row, "entity_id")
            country = normalize_country(getattr(row, "country", ""))
            name = getattr(row, "business_name", "")
            address = getattr(row, "business_address", "")

            for key in candidate_block_keys(name, address):
                if country:
                    index[(country, key)].add(entity_id)
                else:
                    index[("", key)].add(entity_id)

    return index


def select_sampled_ground_truth(sample_size, seed=42):
    """Read only the relevant ground-truth rows and sample a manageable subset."""
    df = pd.read_csv(
        GROUND_TRUTH_PATH,
        sep="\t",
        dtype=str,
        usecols=["source1_entity_id", "matched_entity_ids"],
    )

    df = df[df["matched_entity_ids"].fillna("").str.strip() != ""].copy()

    if sample_size <= 0:
        return df

    if sample_size >= len(df):
        return df

    return df.sample(n=sample_size, random_state=seed).reset_index(drop=True)


def fetch_sampled_s1_rows(sample_ids):
    """Read only the sampled S1 rows from Source 1 once."""
    sample_ids = set(sample_ids)
    records = []

    for chunk in pd.read_csv(SOURCE1_PATH, sep="\t", dtype=str, chunksize=500_000):
        subset = chunk[chunk["entity_id"].isin(sample_ids)][["entity_id", "business_name", "business_address", "country"]]
        if not subset.empty:
            records.append(subset)

    if not records:
        return pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])

    return pd.concat(records, ignore_index=True)


def generate_candidates_for_record(record, source_index):
    """Generate blocking candidates from one row against a source index."""
    country = normalize_country(record.get("country", ""))
    candidates = set()

    for key in candidate_block_keys(record.get("business_name", ""), record.get("business_address", "")):
        if country:
            candidates.update(source_index.get((country, key), set()))
        else:
            candidates.update(source_index.get(("", key), set()))

    return candidates


def main():
    parser = argparse.ArgumentParser(description="Evaluate entity-resolution blocking recall.")
    parser.add_argument("--sample-size", type=int, default=200, help="Number of Source 1 entities to sample from the train ground truth.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    args = parser.parse_args()

    gt_sample = select_sampled_ground_truth(args.sample_size, seed=args.seed)
    print("=" * 80)
    print("BLOCKING RECALL EVALUATION")
    print("=" * 80)
    print(f"Sampled S1 rows: {len(gt_sample):,}")

    source2_index = build_source_index(SOURCE2_PATH)
    source3_index = build_source_index(SOURCE3_PATH)
    print(f"Source 2 index blocks: {len(source2_index):,}")
    print(f"Source 3 index blocks: {len(source3_index):,}")

    sampled_ids = set(gt_sample["source1_entity_id"])
    s1_rows = fetch_sampled_s1_rows(sampled_ids)
    s1_lookup = s1_rows.set_index("entity_id").to_dict("index")

    total_true_s2 = 0
    total_true_s3 = 0
    total_true_all = 0
    retrieved_s2 = 0
    retrieved_s3 = 0
    retrieved_all = 0
    zero_candidate_rows = 0
    candidate_counts = []
    missed_examples = []

    for row in gt_sample.itertuples(index=False):
        s1_id = getattr(row, "source1_entity_id")
        true_ids = parse_match_ids(getattr(row, "matched_entity_ids"))
        if not true_ids:
            continue

        true_s2 = {item for item in true_ids if item.startswith("S2-")}
        true_s3 = {item for item in true_ids if item.startswith("S3-")}

        total_true_s2 += len(true_s2)
        total_true_s3 += len(true_s3)
        total_true_all += len(true_ids)

        record = s1_lookup.get(s1_id)
        if record is None:
            candidates = set()
        else:
            candidates = set()
            candidates.update(generate_candidates_for_record(record, source2_index))
            candidates.update(generate_candidates_for_record(record, source3_index))

        candidate_counts.append(len(candidates))
        if len(candidates) == 0:
            zero_candidate_rows += 1

        retrieved_s2_ids = true_s2 & candidates
        retrieved_s3_ids = true_s3 & candidates
        retrieved_s2 += len(retrieved_s2_ids)
        retrieved_s3 += len(retrieved_s3_ids)
        retrieved_all += len((set(true_ids) & candidates))

        for source_name, source_matches in [("S2", true_s2), ("S3", true_s3)]:
            source_index = source2_index if source_name == "S2" else source3_index
            for match_id in sorted(source_matches - (set(candidates) & source_matches)):
                if len(missed_examples) < 10:
                    row_detail = s1_lookup.get(s1_id)
                    missed_examples.append({
                        "source1_id": s1_id,
                        "source": source_name,
                        "missed_match": match_id,
                        "name": row_detail.get("business_name", "") if row_detail else "",
                        "address": row_detail.get("business_address", "") if row_detail else "",
                        "country": row_detail.get("country", "") if row_detail else "",
                    })
                else:
                    break

    def safe_recall(num_retrieved, num_true):
        return 0.0 if num_true == 0 else num_retrieved / num_true

    avg_candidates = sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0.0
    max_candidates = max(candidate_counts) if candidate_counts else 0

    print("\nRecall summary")
    print("-" * 80)
    print(f"Source 2 recall: {safe_recall(retrieved_s2, total_true_s2):.4f} ({retrieved_s2:,}/{total_true_s2:,})")
    print(f"Source 3 recall: {safe_recall(retrieved_s3, total_true_s3):.4f} ({retrieved_s3:,}/{total_true_s3:,})")
    print(f"Overall recall: {safe_recall(retrieved_all, total_true_all):.4f} ({retrieved_all:,}/{total_true_all:,})")
    print(f"Average candidate count per S1: {avg_candidates:.2f}")
    print(f"Maximum candidate count: {max_candidates:,}")
    print(f"S1 records with zero candidates: {zero_candidate_rows:,}")

    print("\nExamples of true matches missed by the blocker")
    print("-" * 80)
    if missed_examples:
        for example in missed_examples:
            print(
                f"S1={example['source1_id']} | source={example['source']} | missed={example['missed_match']} | "
                f"country={example['country']} | name={example['name'][:120]} | address={example['address'][:120]}"
            )
    else:
        print("No missed true matches found in this sample.")


if __name__ == "__main__":
    main()
