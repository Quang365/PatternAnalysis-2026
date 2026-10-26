"""
Training entry point for FLAN-T5 radiology report summarisation.

Supports two fine-tuning methods:
1. Full fine-tuning
2. LoRA fine-tuning

This stage implements training configuration and model selection.
The custom PyTorch training loop will be added in the next stage.
"""

import argparse
import random

import numpy as np
import torch

from dataset import MODEL_NAME
from modules import load_full_model, load_lora_model, inspect_model


def parse_args():
    """
    Parse command-line arguments for model training.

    Returns:
        argparse.Namespace: Training configuration.
    """

    parser = argparse.ArgumentParser(
        description="Train FLAN-T5 for radiology report summarisation."
    )

    # Select the fine-tuning approach.
    parser.add_argument(
        "--method",
        type=str,
        choices=["full", "lora"],
        required=True,
        help="Fine-tuning method: full or lora.",
    )

    # Training hyperparameters.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # LoRA-specific hyperparameters.
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Directory for saving training outputs.
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
    )

    return parser.parse_args()


def set_seed(seed):
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): Random seed.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """
    Select the available computation device.

    Returns:
        torch.device: CUDA device if available, otherwise CPU.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def initialise_model(args):
    """
    Initialise the selected fine-tuning model.

    Args:
        args (argparse.Namespace): Training configuration.

    Returns:
        torch.nn.Module: Initialised model.
    """

    if args.method == "full":
        model = load_full_model()

    else:
        model = load_lora_model(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )

    return model


def main():
    """Initialise the training configuration and selected model."""

    args = parse_args()

    # Validate configuration before loading the model.
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")

    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")

    if args.learning_rate is not None and args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    if args.lora_rank <= 0:
        raise ValueError("lora_rank must be positive")

    if args.lora_alpha <= 0:
        raise ValueError("lora_alpha must be positive")

    if not 0 <= args.lora_dropout < 1:
        raise ValueError("lora_dropout must be in [0, 1)")

    # Use method-specific initial learning rates.
    # These are starting configurations, not tuned values.
    if args.learning_rate is None:
        args.learning_rate = 5e-5 if args.method == "full" else 1e-4

    set_seed(args.seed)

    device = get_device()

    print("\n=== Training Configuration ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Method: {args.method}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Random seed: {args.seed}")
    print(f"Output directory: {args.output_dir}")

    # Load the selected model without starting training.
    model = initialise_model(args)

    inspect_model(
        model,
        "Full Fine-Tuning" if args.method == "full" else "LoRA Fine-Tuning",
    )

    print("\nTraining configuration initialised successfully.")
    print("Training loop has not been implemented yet.")


if __name__ == "__main__":
    main()