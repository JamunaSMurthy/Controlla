"""Text encoders for Controlla.

Backends:
- simple: lightweight deterministic hash-token encoder for tests and smoke runs.
- hf_clip: frozen Hugging Face CLIP text encoder for paper-aligned experiments.

The simple backend is not intended to replace CLIP for final experiments. It is
only a reproducible local fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch
from torch import nn


@dataclass
class TextEncoderOutput:
    """Container for text features.

    Attributes:
        token_embeddings: Tensor with shape [B, T, D].
        pooled_embedding: Tensor with shape [B, D].
        attention_mask: Float tensor with shape [B, T].
    """

    token_embeddings: torch.Tensor
    pooled_embedding: torch.Tensor
    attention_mask: torch.Tensor


class TextEncoder(nn.Module):
    """Text encoder with lightweight and frozen-CLIP backends."""

    def __init__(
        self,
        hidden_dim: int,
        max_tokens: int,
        backend: str = "simple",
        vocab_size: int = 8192,
        pretrained_name: str = "openai/clip-vit-base-patch32",
        local_files_only: bool = True,
        freeze: bool = True,
        dropout: float = 0.1,
        num_heads: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        if vocab_size <= 1:
            raise ValueError(f"vocab_size must be > 1, got {vocab_size}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.backend = backend
        self.hidden_dim = hidden_dim
        self.max_tokens = max_tokens
        self.vocab_size = vocab_size

        if self.backend == "simple":
            self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
            self.position_embedding = nn.Embedding(max_tokens, hidden_dim)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_norm = nn.LayerNorm(hidden_dim)

        elif self.backend == "hf_clip":
            try:
                from transformers import CLIPTextModel, CLIPTokenizer
            except ImportError as error:
                raise ImportError(
                    "hf_clip text encoder requested but transformers is not installed. "
                    "Install with: pip install transformers"
                ) from error

            self.tokenizer = CLIPTokenizer.from_pretrained(
                pretrained_name,
                local_files_only=local_files_only,
            )
            self.encoder = CLIPTextModel.from_pretrained(
                pretrained_name,
                local_files_only=local_files_only,
            )

            if freeze:
                self.encoder.requires_grad_(False)

            clip_dim = int(self.encoder.config.hidden_size)

            self.output_projection = nn.Sequential(
                nn.LayerNorm(clip_dim),
                nn.Linear(clip_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )

            self.token_projection = nn.Sequential(
                nn.LayerNorm(clip_dim),
                nn.Linear(clip_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )

        else:
            raise ValueError(f"Unsupported text encoder backend: {backend}")

    @staticmethod
    def _stable_hash_token(word: str, vocab_size: int) -> int:
        """Map a word to a deterministic token id in [1, vocab_size - 1]."""
        digest = hashlib.md5(word.encode("utf-8")).hexdigest()
        value = int(digest, 16)
        return value % (vocab_size - 1) + 1

    def _simple_tokenize(
        self,
        prompts: list[str],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids = torch.zeros(
            (len(prompts), self.max_tokens),
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros(
            (len(prompts), self.max_tokens),
            dtype=torch.float32,
            device=device,
        )

        for row_index, prompt in enumerate(prompts):
            if prompt is None:
                prompt = ""

            words = str(prompt).lower().strip().split()[: self.max_tokens]

            for token_index, word in enumerate(words):
                token_ids[row_index, token_index] = self._stable_hash_token(
                    word,
                    self.vocab_size,
                )
                attention_mask[row_index, token_index] = 1.0

        return token_ids, attention_mask

    def _forward_simple(self, prompts: list[str]) -> TextEncoderOutput:
        device = self.token_embedding.weight.device

        token_ids, attention_mask = self._simple_tokenize(prompts, device=device)

        positions = torch.arange(self.max_tokens, device=device).unsqueeze(0)
        positions = positions.expand_as(token_ids)

        embeddings = self.token_embedding(token_ids) + self.position_embedding(positions)
        embeddings = embeddings * attention_mask.unsqueeze(-1)

        encoded = self.encoder(
            embeddings,
            src_key_padding_mask=attention_mask.eq(0.0),
        )
        encoded = self.output_norm(encoded)
        encoded = encoded * attention_mask.unsqueeze(-1)

        pooled = encoded.sum(dim=1)
        pooled = pooled / attention_mask.sum(dim=1, keepdim=True).clamp_min(1.0)

        return TextEncoderOutput(
            token_embeddings=encoded,
            pooled_embedding=pooled,
            attention_mask=attention_mask,
        )

    def _forward_hf_clip(self, prompts: list[str]) -> TextEncoderOutput:
        device = self.output_projection[1].weight.device

        tokenizer_output = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        )
        tokenizer_output = {
            key: value.to(device)
            for key, value in tokenizer_output.items()
        }

        # If the CLIP encoder is frozen, no_grad saves memory. If unfrozen,
        # gradients still flow.
        any_trainable = any(parameter.requires_grad for parameter in self.encoder.parameters())

        if any_trainable:
            encoder_output = self.encoder(**tokenizer_output)
        else:
            with torch.no_grad():
                encoder_output = self.encoder(**tokenizer_output)

        token_hidden = encoder_output.last_hidden_state
        pooled_hidden = encoder_output.pooler_output

        token_embeddings = self.token_projection(token_hidden)
        pooled_embedding = self.output_projection(pooled_hidden)

        attention_mask = tokenizer_output["attention_mask"].float()
        token_embeddings = token_embeddings * attention_mask.unsqueeze(-1)

        return TextEncoderOutput(
            token_embeddings=token_embeddings,
            pooled_embedding=pooled_embedding,
            attention_mask=attention_mask,
        )

    def forward(self, prompts: list[str]) -> TextEncoderOutput:
        """Encode a batch of prompt strings into token and pooled embeddings."""

        if not isinstance(prompts, list):
            raise TypeError(f"prompts must be a list[str], got {type(prompts)}")

        if len(prompts) == 0:
            raise ValueError("prompts must contain at least one string")

        if self.backend == "simple":
            return self._forward_simple(prompts)

        if self.backend == "hf_clip":
            return self._forward_hf_clip(prompts)

        raise RuntimeError(f"Unexpected text encoder backend: {self.backend}")