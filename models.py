"""
Neural modules for the semantic-spatial LDTO backbone.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class MLPBlock(nn.Module):
    """A simple feed-forward block used repeatedly across the model."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialEncoder(nn.Module):
    """Encoder for structured spatial-demographic features."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.backbone = nn.Sequential(
            MLPBlock(config.spatial_dim, config.hidden_dim, config.dropout),
            MLPBlock(config.hidden_dim, config.hidden_dim, config.dropout),
        )
        self.projector = nn.Linear(config.hidden_dim, config.projection_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(x)
        projection = F.normalize(self.projector(hidden), dim=-1)
        return hidden, projection


class SemanticEncoder(nn.Module):
    """Encoder for policy semantic embeddings."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.backbone = nn.Sequential(
            MLPBlock(config.semantic_dim, config.hidden_dim, config.dropout),
            MLPBlock(config.hidden_dim, config.hidden_dim, config.dropout),
        )
        self.projector = nn.Linear(config.hidden_dim, config.projection_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(x)
        projection = F.normalize(self.projector(hidden), dim=-1)
        return hidden, projection


class SemanticSpatialFusion(nn.Module):
    """Collaborative fusion module for spatial and semantic branches."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        fusion_input = config.hidden_dim * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, config.fusion_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.fusion_dim),
            nn.Linear(config.fusion_dim, config.fusion_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.fusion_dim),
        )

    def forward(self, spatial_hidden: torch.Tensor, semantic_hidden: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([spatial_hidden, semantic_hidden], dim=-1)
        return self.fusion(fused)


class SuitabilityHead(nn.Module):
    """Regression head that predicts the facility suitability score."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(config.fusion_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        # The sigmoid function keeps the prediction in the normalized [0, 1] range.
        return torch.sigmoid(self.head(fused)).squeeze(-1)


class LDTOModel(nn.Module):
    """End-to-end semantic-spatial prediction model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.spatial_encoder = SpatialEncoder(config)
        self.semantic_encoder = SemanticEncoder(config)
        self.fusion = SemanticSpatialFusion(config)
        self.predictor = SuitabilityHead(config)

    def encode(self, spatial: torch.Tensor, semantic: torch.Tensor) -> Dict[str, torch.Tensor]:
        spatial_hidden, spatial_proj = self.spatial_encoder(spatial)
        semantic_hidden, semantic_proj = self.semantic_encoder(semantic)
        fused = self.fusion(spatial_hidden, semantic_hidden)
        return {
            "spatial_hidden": spatial_hidden,
            "semantic_hidden": semantic_hidden,
            "spatial_proj": spatial_proj,
            "semantic_proj": semantic_proj,
            "fused": fused,
        }

    def forward(self, spatial: torch.Tensor, semantic: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = self.encode(spatial, semantic)
        score = self.predictor(encoded["fused"])
        encoded["score"] = score
        return encoded
