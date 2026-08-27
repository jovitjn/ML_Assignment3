"""Reusable model, metadata, and decoding utilities for next-word prediction."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Sequence

import torch
from torch import Tensor, nn


TOKEN_PATTERN = re.compile(r"[^a-zA-Z0-9\s]")


@dataclass(frozen=True)
class LanguageModelMetadata:
    """Vocabulary and architecture information stored separately from weights."""

    stoi: dict[str, int]
    itos: tuple[str, ...]
    block_size: int
    embedding_dim: int = 64
    hidden_dim: int = 128
    pad_token: str = "<PAD>"
    unk_token: str | None = None

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def pad_index(self) -> int:
        return self.stoi[self.pad_token]

    @property
    def unknown_index(self) -> int:
        if self.unk_token is not None and self.unk_token in self.stoi:
            return self.stoi[self.unk_token]
        return self.pad_index

    @classmethod
    def load(cls, path: str | Path) -> "LanguageModelMetadata":
        with Path(path).open(encoding="utf-8") as handle:
            raw = json.load(handle)
        itos = tuple(raw["itos"])
        stoi = raw.get("stoi") or {word: index for index, word in enumerate(itos)}

        metadata = cls(
            stoi={word: int(index) for word, index in stoi.items()},
            itos=itos,
            block_size=int(raw["block_size"]),
            embedding_dim=int(raw.get("embedding_dim", 64)),
            hidden_dim=int(raw.get("hidden_dim", 128)),
            pad_token=raw.get("pad_token", "<PAD>"),
            unk_token=raw.get("unk_token"),
        )
        metadata.validate()
        return metadata

    def save(self, path: str | Path) -> None:
        payload = {
            "block_size": self.block_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "itos": list(self.itos),
        }
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    def validate(self) -> None:
        if self.block_size < 1:
            raise ValueError("block_size must be positive")
        if len(self.stoi) != len(self.itos):
            raise ValueError("stoi and itos must contain the same number of tokens")
        for index, word in enumerate(self.itos):
            if self.stoi.get(word) != index:
                raise ValueError("stoi and itos are not inverse mappings")
        if self.pad_token not in self.stoi:
            raise ValueError(f"missing pad token: {self.pad_token}")


class NextWordPredictor(nn.Module):
    """Fixed-context feed-forward neural language model."""

    def __init__(
        self,
        block_size: int,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden = nn.Linear(block_size * embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.ndim != 2 or token_ids.shape[1] != self.block_size:
            raise ValueError(
                f"expected token_ids with shape (batch, {self.block_size}), "
                f"received {tuple(token_ids.shape)}"
            )
        embedded = self.embedding(token_ids).flatten(start_dim=1)
        return self.output(torch.relu(self.hidden(embedded)))


def normalize_tokens(text: str, minimum_length: int = 3) -> list[str]:
    """Match the lowercase alphanumeric preprocessing used by the notebook."""

    cleaned = TOKEN_PATTERN.sub(" ", text).lower()
    return [token for token in cleaned.split() if len(token) >= minimum_length]


def encode_context(
    tokens: Sequence[str], metadata: LanguageModelMetadata
) -> list[int]:
    """Encode, left-pad, and truncate a token sequence to one context window."""

    encoded = [metadata.stoi.get(token, metadata.unknown_index) for token in tokens]
    encoded = encoded[-metadata.block_size :]
    padding = [metadata.pad_index] * (metadata.block_size - len(encoded))
    return padding + encoded


def choose_next_token(
    logits: Tensor,
    *,
    greedy: bool = False,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> int:
    """Choose one token with greedy or temperature/top-k sampling."""

    if logits.ndim == 2:
        if logits.shape[0] != 1:
            raise ValueError("batched decoding currently expects a batch size of one")
        logits = logits.squeeze(0)
    if logits.ndim != 1:
        raise ValueError("logits must have shape (vocab,) or (1, vocab)")
    if greedy:
        return int(torch.argmax(logits).item())
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")

    scaled = logits / temperature
    if top_k is not None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        top_k = min(top_k, scaled.numel())
        values, indices = torch.topk(scaled, top_k)
        selected = torch.multinomial(
            torch.softmax(values, dim=-1), 1, generator=generator
        )
        return int(indices[selected].item())

    return int(
        torch.multinomial(
            torch.softmax(scaled, dim=-1), 1, generator=generator
        ).item()
    )


def generate_tokens(
    model: NextWordPredictor,
    metadata: LanguageModelMetadata,
    prompt_tokens: Sequence[str],
    num_tokens: int,
    *,
    greedy: bool = False,
    temperature: float = 1.0,
    top_k: int | None = None,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> list[str]:
    """Autoregressively extend a tokenized prompt."""

    generated = list(prompt_tokens)
    model.eval()
    with torch.inference_mode():
        for _ in range(num_tokens):
            context = torch.tensor(
                [encode_context(generated, metadata)],
                dtype=torch.long,
                device=device,
            )
            logits = model(context).clone()
            logits[..., metadata.pad_index] = -torch.inf
            if metadata.unk_token is not None:
                logits[..., metadata.unknown_index] = -torch.inf
            next_index = choose_next_token(
                logits,
                greedy=greedy,
                temperature=temperature,
                top_k=top_k,
                generator=generator,
            )
            generated.append(metadata.itos[next_index])
    return generated


def load_model(
    checkpoint_path: str | Path,
    metadata: LanguageModelMetadata,
    device: torch.device | str,
) -> NextWordPredictor:
    """Construct a model and load a state-dict-only checkpoint."""

    model = NextWordPredictor(
        block_size=metadata.block_size,
        vocab_size=metadata.vocab_size,
        embedding_dim=metadata.embedding_dim,
        hidden_dim=metadata.hidden_dim,
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Preserve compatibility with the layer names in the original notebook.
    legacy_names = {"emb.weight", "lin1.weight", "lin1.bias", "lin2.weight", "lin2.bias"}
    if legacy_names.issubset(state_dict):
        state_dict = {
            key.replace("emb.", "embedding.")
            .replace("lin1.", "hidden.")
            .replace("lin2.", "output."): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict)
    model.eval()
    return model
