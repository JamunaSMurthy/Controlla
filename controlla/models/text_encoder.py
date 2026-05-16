"""Text encoders for Controlla.

The default backend is a lightweight hash-token encoder so tests and dummy runs do
not require external checkpoint downloads. A frozen Hugging Face CLIP backend can
be enabled later for paper experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TextEncoderOutput:
    """Container for text features.

    Attributes:
        token_embeddings: Tensor with shape `[B, T, D]`.
        pooled_embedding: Tensor with shape `[B, D]`.
        attention_mask: Float tensor with shape `[B, T]`.
    """

    token_embeddings: torch.Tensor
    pooled_embedding: torch.Tensor
    attention_mask: torch.Tensor


class TextEncoder(nn.Module):
    """Minimal text encoder with an optional frozen CLIP backend."""

    def __init__(
        self,
        hidden_dim: int,
        max_tokens: int,
        backend: str = "simple",
        vocab_size: int = 8192,
        pretrained_name: str = "openai/clip-vit-base-patch32",
    ) -> None:
        super().__init__()
        self.backend = backend
        self.hidden_dim = hidden_dim
        self.max_tokens = max_tokens
        self.vocab_size = vocab_size

        if self.backend == "simple":
            self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
            self.position_embedding = nn.Embedding(max_tokens, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        elif self.backend == "hf_clip":
            from transformers import CLIPTextModel, CLIPTokenizer

            self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_name)
            self.encoder = CLIPTextModel.from_pretrained(pretrained_name)
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
            self.output_projection = nn.Linear(self.encoder.config.hidden_size, hidden_dim)
        else:
            raise ValueError(f"Unsupported text encoder backend: {backend}")

    def _simple_tokenize(self, prompts: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids = torch.zeros((len(prompts), self.max_tokens), dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(prompts), self.max_tokens), dtype=torch.float32, device=device)
        for row_index, prompt in enumerate(prompts):
            words = prompt.lower().split()[: self.max_tokens]
            for token_index, word in enumerate(words):
                token_ids[row_index, token_index] = abs(hash(word)) % (self.vocab_size - 1) + 1
                attention_mask[row_index, token_index] = 1.0
        return token_ids, attention_mask

    def forward(self, prompts: list[str]) -> TextEncoderOutput:
        """Encode a batch of prompt strings into token and pooled embeddings."""
        if self.backend == "simple":
            device = self.token_embedding.weight.device
            token_ids, attention_mask = self._simple_tokenize(prompts, device=device)
            positions = torch.arange(self.max_tokens, device=device).unsqueeze(0).expand_as(token_ids)
            embeddings = self.token_embedding(token_ids) + self.position_embedding(positions)
            encoded = self.encoder(embeddings, src_key_padding_mask=attention_mask.eq(0.0))
            pooled = (encoded * attention_mask.unsqueeze(-1)).sum(dim=1)
            pooled = pooled / attention_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            return TextEncoderOutput(token_embeddings=encoded, pooled_embedding=pooled, attention_mask=attention_mask)

        tokenizer_output = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        )
        tokenizer_output = {key: value.to(self.output_projection.weight.device) for key, value in tokenizer_output.items()}
        with torch.no_grad():
            encoder_output = self.encoder(**tokenizer_output)
        hidden = self.output_projection(encoder_output.last_hidden_state)
        pooled = hidden[:, 0]
        attention_mask = tokenizer_output["attention_mask"].float()
        return TextEncoderOutput(token_embeddings=hidden, pooled_embedding=pooled, attention_mask=attention_mask)