"""Command-line inference for the trained next-word model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import LanguageModelMetadata, generate_tokens, load_model, normalize_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained model.")
    parser.add_argument("prompt", help="starting text")
    parser.add_argument("--checkpoint", type=Path, default=Path("model.pth"))
    parser.add_argument("--metadata", type=Path, default=Path("metadata.json"))
    parser.add_argument("--num-words", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_words < 1:
        raise ValueError("num_words must be positive")
    prompt_tokens = normalize_tokens(args.prompt)
    if not prompt_tokens:
        raise ValueError("prompt must contain at least one alphanumeric word of length >= 3")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = LanguageModelMetadata.load(args.metadata)
    model = load_model(args.checkpoint, metadata, device)
    generator = torch.Generator(device=device.type).manual_seed(args.seed)
    generated = generate_tokens(
        model,
        metadata,
        prompt_tokens,
        args.num_words,
        greedy=args.greedy,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
        generator=generator,
    )
    print(" ".join(generated))


if __name__ == "__main__":
    main()
