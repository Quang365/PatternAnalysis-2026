"""Dataset loading and inspection for radiology report summarisation."""

from datasets import load_dataset
from transformers import AutoTokenizer


DATASET_NAME = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"

MODEL_NAME = "google/flan-t5-base"

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128

PROMPT = (
    "Summarize the following radiology report in simple language "
    "that a patient without medical knowledge can understand:\n\n"
)

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

def load_tokenizer():
    """Load the tokenizer associated with FLAN-T5 Base."""
    return AutoTokenizer.from_pretrained(MODEL_NAME)

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

def preprocess_function(examples, tokenizer):
    """Tokenize radiology reports and reference summaries."""

    inputs = [
        PROMPT + report
        for report in examples["radiology_report"]
    ]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
    )

    labels = tokenizer(
        text_target=examples["layman_report"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

def inspect_preprocessing(dataset, tokenizer):
    """Inspect tokenization results on a small training sample."""

    sample = dataset["train"].select(range(3))

    examples = {
        "radiology_report": list(sample["radiology_report"]),
        "layman_report": list(sample["layman_report"]),
    }

    tokenized = preprocess_function(
        examples,
        tokenizer,
    )

    print("\n=== Preprocessing Inspection ===")

    for i in range(len(sample)):
        input_ids = tokenized["input_ids"][i]
        label_ids = tokenized["labels"][i]

        print(f"\nSample {i + 1}")
        print(f"Input tokens: {len(input_ids)}")
        print(f"Target tokens: {len(label_ids)}")

        print("\nDecoded input:")
        print(tokenizer.decode(input_ids, skip_special_tokens=True)[:400])

        print("\nDecoded target:")
        print(tokenizer.decode(label_ids, skip_special_tokens=True)[:400])

        assert len(input_ids) <= MAX_INPUT_LENGTH
        assert len(label_ids) <= MAX_TARGET_LENGTH

    print("\nPreprocessing inspection completed successfully.")

if __name__ == "__main__":
    dataset = load_radiology_dataset()

    inspect_dataset(dataset)

    tokenizer = load_tokenizer()

    inspect_preprocessing(dataset, tokenizer)