"""
Inference pipeline for FLAN-T5 radiology report summarisation.

Supports full fine-tuning and LoRA fine-tuning checkpoints.
"""

import argparse

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

from dataset import MODEL_NAME, PROMPT


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
        required=True,
        help="Radiology report to summarise.",
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


def main():
    """Run inference from the command line."""
    args = parse_args()

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