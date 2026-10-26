"""Dataset loading and inspection for radiology report summarisation."""

from datasets import load_dataset


DATASET_NAME = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"

EXPECTED_SPLITS = {
    "train": 150454,
    "validation": 10000,
    "test": 10537,
}

REQUIRED_COLUMNS = {
    "source",
    "images_path",
    "radiology_report",
    "layman_report",
}


def load_radiology_dataset():
    """Load the official BioLaySumm2025 LaymanRRG dataset."""
    return load_dataset(DATASET_NAME)


def inspect_dataset(dataset):
    """Inspect dataset splits, columns, and sample records."""

    print("\n=== Dataset Inspection ===")

    for split, expected_size in EXPECTED_SPLITS.items():
        if split not in dataset:
            raise ValueError(f"Missing dataset split: {split}")

        split_data = dataset[split]

        print(f"\nSplit: {split}")
        print(f"Samples: {len(split_data):,}")
        print(f"Columns: {split_data.column_names}")

        if len(split_data) != expected_size:
            raise ValueError(
                f"Unexpected size for {split}: "
                f"{len(split_data)} != {expected_size}"
            )

        missing_columns = REQUIRED_COLUMNS - set(split_data.column_names)

        if missing_columns:
            raise ValueError(
                f"Missing columns in {split}: {missing_columns}"
            )

        if len(split_data) > 0:
            sample = split_data[0]

            print("Sample radiology report:")
            print(str(sample["radiology_report"])[:300])

            print("\nSample layperson summary:")
            print(str(sample["layman_report"])[:300])

    print("\nDataset inspection completed successfully.")


if __name__ == "__main__":
    dataset = load_radiology_dataset()
    inspect_dataset(dataset)