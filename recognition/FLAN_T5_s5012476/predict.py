"""
Inference pipeline for FLAN-T5 radiology report summarisation.

Supports full fine-tuning and LoRA fine-tuning checkpoints.
"""

import argparse
import csv
import json

import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from rouge_score import rouge_scorer
from pathlib import Path

from dataset import MODEL_NAME, PROMPT, load_radiology_dataset

def load_evaluation_dataset(max_eval_samples=0):
    """
    Load the official validation split for model evaluation.

    Args:
        max_eval_samples (int): Maximum samples to evaluate.
            A value of 0 uses the entire validation split.

    Returns:
        Dataset: Validation dataset.
    """
    dataset = load_radiology_dataset()
    validation_dataset = dataset["validation"]

    if max_eval_samples > 0:
        sample_count = min(
            max_eval_samples,
            len(validation_dataset),
        )

        validation_dataset = validation_dataset.select(
            range(sample_count)
        )

    print("\n=== Evaluation Dataset ===")
    print(f"Validation samples: {len(validation_dataset):,}")
    print(f"Columns: {validation_dataset.column_names}")

    return validation_dataset

def parse_args():
    """Parse inference arguments."""
    parser = argparse.ArgumentParser(
        description="Generate layman summaries using fine-tuned FLAN-T5."
    )

    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["full", "lora"],
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the fine-tuned checkpoint.",
    )

    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Radiology report for single prediction.",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the model on the validation dataset.",
    )

    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=0,
        help="Maximum evaluation samples; 0 uses the full validation split.",
    )

    return parser.parse_args()

def load_model_and_tokenizer(method, checkpoint, device):
    """Load the fine-tuned model and its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    if method == "full":
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    else:
        base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

        model = PeftModel.from_pretrained(
            base_model,
            checkpoint,
        )

    model = model.to(device)
    model.eval()

    return model, tokenizer

def generate_summary(
    model,
    tokenizer,
    report,
    device,
    max_new_tokens=128,
    num_beams=4,
):
    """Generate a patient-friendly summary of a radiology report."""
    input_text = PROMPT + report

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
        )

    summary = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    )

    return summary

def calculate_rouge(predictions):
    """
    Calculate average ROUGE F1 scores across evaluation samples.

    Args:
        predictions (list): Generated predictions and references.

    Returns:
        dict: Average ROUGE scores.
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"],
        use_stemmer=True,
    )

    scores = {
        "rouge1": [],
        "rouge2": [],
        "rougeL": [],
        "rougeLsum": [],
    }

    for prediction in predictions:
        reference = prediction["reference_summary"]
        generated = prediction["generated_summary"]

        result = scorer.score(reference, generated)

        for metric in scores:
            scores[metric].append(result[metric].fmeasure)

    return {
        metric: float(np.mean(values))
        for metric, values in scores.items()
    }

def plot_rouge_scores(rouge_scores, output_dir, method):
    """
    Generate a bar chart of ROUGE F1 scores.

    Args:
        rouge_scores (dict): ROUGE evaluation results.
        output_dir (Path): Directory for evaluation outputs.
        method (str): Fine-tuning method.
    """
    metrics = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "ROUGE-Lsum"]

    values = [rouge_scores[metric] for metric in metrics]

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(labels, values)

    ax.set_title(f"{method.upper()} — ROUGE Evaluation")
    ax.set_ylabel("ROUGE F1 Score")
    ax.set_ylim(0, 1)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    plt.tight_layout()

    figure_path = output_dir / "rouge_scores.png"

    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"ROUGE figure saved to: {figure_path}")

def evaluate_predictions(
    model,
    tokenizer,
    validation_dataset,
    device,
    args,
):
    """
    Generate and save predictions for the validation dataset.

    Args:
        model: Fine-tuned FLAN-T5 model.
        tokenizer: FLAN-T5 tokenizer.
        validation_dataset: Official validation dataset.
        device (torch.device): Computation device.
        args (argparse.Namespace): Inference configuration.
    """
    output_dir = Path("outputs") / args.method / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = []

    print("\n=== Generating Evaluation Predictions ===")

    for index, example in enumerate(validation_dataset):
        report = example["radiology_report"]
        reference = example["layman_report"]

        generated_summary = generate_summary(
            model=model,
            tokenizer=tokenizer,
            report=report,
            device=device,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )

        predictions.append(
            {
                "sample_id": index,
                "source": example["source"],
                "radiology_report": report,
                "reference_summary": reference,
                "generated_summary": generated_summary,
            }
        )

        print(
            f"Generated {index + 1}/{len(validation_dataset)} summaries"
        )

    csv_path = output_dir / "predictions.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample_id",
                "source",
                "radiology_report",
                "reference_summary",
                "generated_summary",
            ],
        )

        writer.writeheader()
        writer.writerows(predictions)

    print(f"\nPredictions saved to: {csv_path}")

    # Calculate ROUGE scores for the generated summaries.
    rouge_scores = calculate_rouge(predictions)

    rouge_path = output_dir / "rouge_scores.json"

    with open(rouge_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "method": args.method,
                "num_samples": len(predictions),
                "scores": rouge_scores,
            },
            file,
            indent=4,
        )

    print("\n=== ROUGE Evaluation ===")

    for metric, score in rouge_scores.items():
        print(f"{metric}: {score:.4f}")

    print(f"ROUGE scores saved to: {rouge_path}")

    plot_rouge_scores(
        rouge_scores=rouge_scores,
        output_dir=output_dir,
        method=args.method,
    )

    return predictions

def main():
    """Run inference from the command line."""
    args = parse_args()

    if args.max_eval_samples < 0:
        raise ValueError("max_eval_samples must be non-negative.")

    if args.evaluate and args.text is not None:
        raise ValueError(
            "Use either --evaluate or --text, not both."
        )

    if not args.evaluate and not args.text:
        raise ValueError(
            "Provide --text for prediction or --evaluate for evaluation."
        )
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")

    if args.num_beams <= 0:
        raise ValueError("num_beams must be positive.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n=== Inference Configuration ===")
    print(f"Method: {args.method}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")

    model, tokenizer = load_model_and_tokenizer(
        method=args.method,
        checkpoint=args.checkpoint,
        device=device,
    )

    if args.evaluate:
        validation_dataset = load_evaluation_dataset(
            max_eval_samples=args.max_eval_samples
        )

        evaluate_predictions(
            model=model,
            tokenizer=tokenizer,
            validation_dataset=validation_dataset,
            device=device,
            args=args,
        )

        print("\nEvaluation predictions completed successfully.")
        return

    summary = generate_summary(
        model=model,
        tokenizer=tokenizer,
        report=args.text,
        device=device,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )

    print("\n=== Radiology Report ===")
    print(args.text)

    print("\n=== Generated Layman Summary ===")
    print(summary)


if __name__ == "__main__":
    main()