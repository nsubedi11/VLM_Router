# models/resolution_router.py
#
# ResolutionRouter: lightweight query encoder + MLP head that predicts
# pool_level ∈ {0, 1, 2} from the user text query in the input sequence.
#
# Architecture:
#   query_ids  ──► embed_fn (frozen LLM embeddings)        [B, L, D_llm]
#              ──► input_proj: D_llm → hidden_dim           [B, L, hidden_dim]
#              ──► 2-layer TransformerEncoder (Pre-LN)      [B, L, hidden_dim]
#              ──► mean-pool over real tokens               [B, hidden_dim]
#              ──► MLP: hidden_dim → hidden_dim → num_levels
#              ──► logits                                   [B, num_levels]
#
# Usage
# -----
#   from models.resolution_router import ResolutionRouter
#   from models.qwen3_vl import Qwen3VLWithOfflineFeatures, extract_query_ids
#
#   # Attach router after loading the base model
#   model.resolution_router = ResolutionRouter(input_dim=2048).to(model.device)
#
#   # Training: compute loss against pool_level labels
#   query_ids = extract_query_ids(input_ids, cfg.vision_end_token_id)
#   logits    = model.resolution_router(query_ids, model.model.get_input_embeddings())
#   loss      = F.cross_entropy(logits, pool_level_labels)
#
#   # Inference: pool_level is predicted automatically inside model.forward()
#   #   once resolution_router is attached.

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Tuple


class ResolutionRouter(nn.Module):
    """
    Predict pool_level ∈ {0, …, num_levels-1} from variable-length query tokens.

    The LLM embedding table (embed_fn) is never updated — gradients from the
    router loss do not propagate through it.

    Args:
        input_dim:   dimension of the LLM token embeddings (e.g. 2048).
        hidden_dim:  internal router width (default 256 keeps it lightweight).
        num_heads:   attention heads in each TransformerEncoderLayer.
        ffn_dim:     feedforward dimension inside the encoder (default 4×hidden).
        num_layers:  number of transformer encoder layers (default 2).
        dropout:     dropout rate for the encoder and MLP head.
        num_levels:  number of pool-level classes (default 3: levels 0, 1, 2).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        ffn_dim: int = 1024,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_levels: int = 3,
    ) -> None:
        super().__init__()

        # Project from LLM embedding space into the lightweight router space
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 2-layer transformer encoder (Pre-LN for training stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLP head: pooled query → pool_level logits
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_levels),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_and_pad(
        self,
        query_ids: List[torch.Tensor],
        embed_fn: nn.Embedding,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad a variable-length list of 1-D query token ID tensors, embed them,
        and return:
            embeds   [B, L, D_llm]  float32
            pad_mask [B, L]         bool   (True = padding, PyTorch convention)
        """
        device = embed_fn.weight.device
        B = len(query_ids)
        max_len = max(max(q.numel() for q in query_ids), 1)

        padded_ids = torch.zeros(B, max_len, dtype=torch.long, device=device)
        pad_mask = torch.ones(B, max_len, dtype=torch.bool, device=device)

        for i, qids in enumerate(query_ids):
            n = qids.numel()
            if n > 0:
                padded_ids[i, :n] = qids.to(device)
                pad_mask[i, :n] = False  # False = real token

        # Embed with frozen LLM weights — no gradient flows back to embed_fn
        with torch.no_grad():
            embeds = embed_fn(padded_ids).float()   # [B, L, D_llm]

        return embeds, pad_mask

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(
        self,
        query_ids: List[torch.Tensor],
        embed_fn: nn.Embedding,
    ) -> torch.Tensor:
        """
        Encode variable-length query tokens to a fixed [B, hidden_dim] vector.

        Pipeline:  embed → input_proj → TransformerEncoder → mean-pool
        """
        embeds, pad_mask = self._embed_and_pad(query_ids, embed_fn)

        # Move to router device in case embed_fn lives on a different device
        router_device = next(self.parameters()).device
        embeds   = embeds.to(router_device)
        pad_mask = pad_mask.to(router_device)

        x = self.input_proj(embeds)                         # [B, L, hidden_dim]
        x = self.encoder(x, src_key_padding_mask=pad_mask)  # [B, L, hidden_dim]

        # Mean-pool over real (non-padding) tokens
        real = ~pad_mask                                     # True = real token
        denom = real.sum(dim=1, keepdim=True).float().clamp(min=1.0)
        pooled = (x * real.unsqueeze(-1).float()).sum(dim=1) / denom  # [B, hidden_dim]

        return pooled

    # ------------------------------------------------------------------
    # Forward / predict
    # ------------------------------------------------------------------

    def forward(
        self,
        query_ids: List[torch.Tensor],
        embed_fn: nn.Embedding,
    ) -> torch.Tensor:
        """
        Args:
            query_ids: list of 1-D LongTensors (one per batch item),
                       as returned by extract_query_ids().
            embed_fn:  model.model.get_input_embeddings() — frozen LLM embedding layer.

        Returns:
            logits: [B, num_levels]  — use F.cross_entropy for training.
        """
        pooled = self.encode(query_ids, embed_fn)
        return self.head(pooled)

    @torch.no_grad()
    def predict(
        self,
        query_ids: List[torch.Tensor],
        embed_fn: nn.Embedding,
    ) -> torch.Tensor:
        """
        Inference-only wrapper around forward().

        Returns:
            pool_levels: [B] int64 tensor — pool_level prediction per batch item.
        """
        logits = self.forward(query_ids, embed_fn)
        return logits.argmax(dim=-1)
