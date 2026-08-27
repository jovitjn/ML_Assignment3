"""Train and evaluate the feed-forward next-word model on a text corpus."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import LanguageModelMetadata, NextWordPredictor, normalize_tokens


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


class NextTokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Sliding fixed-context windows over an encoded token sequence."""

    def __init__(self, token_ids: list[int], block_size: int) -> None:
        if len(token_ids) <= block_size:
            raise ValueError("token sequence must be longer than the context window")
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.token_ids) - self.block_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.token_ids[index : index + self.block_size]
        target = self.token_ids[index + self.block_size]
        return context, target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a fixed-context feed-forward neural language model."
    )
    parser.add_argument("corpus", type=Path, help="UTF-8 plain-text corpus")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--context-length", type=int, default=5)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_tokens(tokens: list[str], validation_fraction: float) -> tuple[list[str], list[str]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    split_index = int(len(tokens) * (1 - validation_fraction))
    return tokens[:split_index], tokens[split_index:]


def build_metadata(
    train_tokens: list[str], block_size: int, embedding_dim: int, hidden_dim: int
) -> LanguageModelMetadata:
    vocabulary = [PAD_TOKEN, UNK_TOKEN, *sorted(set(train_tokens))]
    metadata = LanguageModelMetadata(
        stoi={word: index for index, word in enumerate(vocabulary)},
        itos=tuple(vocabulary),
        block_size=block_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        pad_token=PAD_TOKEN,
        unk_token=UNK_TOKEN,
    )
    metadata.validate()
    return metadata


def encode(tokens: list[str], metadata: LanguageModelMetadata) -> list[int]:
    return [metadata.stoi.get(token, metadata.unknown_index) for token in tokens]


def run_epoch(
    model: NextWordPredictor,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * targets.size(0)
            total_examples += targets.size(0)
    return total_loss / total_examples


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    tokens = normalize_tokens(args.corpus.read_text(encoding="utf-8"))
    train_tokens, validation_tokens = split_tokens(tokens, args.validation_fraction)
    minimum_tokens = args.context_length + 1
    if len(train_tokens) < minimum_tokens or len(validation_tokens) < minimum_tokens:
        raise ValueError("both corpus splits must contain at least context_length + 1 tokens")

    metadata = build_metadata(
        train_tokens, args.context_length, args.embedding_dim, args.hidden_dim
    )
    train_dataset = NextTokenDataset(encode(train_tokens, metadata), metadata.block_size)
    validation_dataset = NextTokenDataset(
        encode(validation_tokens, metadata), metadata.block_size
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NextWordPredictor(
        metadata.block_size,
        metadata.vocab_size,
        metadata.embedding_dim,
        metadata.hidden_dim,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_validation_loss = math.inf
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata.save(args.output_dir / "metadata.json")

    print(f"device={device} vocab_size={metadata.vocab_size}")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        validation_loss = run_epoch(model, validation_loader, criterion, device)
        perplexity = math.exp(min(validation_loss, 20))
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"validation_loss={validation_loss:.4f} "
            f"validation_perplexity={perplexity:.2f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), args.output_dir / "model.pth")

    print(f"saved best checkpoint to {args.output_dir / 'model.pth'}")


if __name__ == "__main__":
    main()
