"""
Loss functions for the LDTO framework.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def prediction_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Supervised regression loss for suitability prediction."""
    return F.mse_loss(prediction, target)


def contrastive_alignment_loss(
    spatial_proj: torch.Tensor,
    semantic_proj: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE loss for semantic-spatial alignment."""
    if spatial_proj.ndim != 2 or semantic_proj.ndim != 2:
        raise ValueError("Contrastive inputs must be rank-2 tensors.")
    logits = spatial_proj @ semantic_proj.t() / temperature
    labels = torch.arange(spatial_proj.size(0), device=spatial_proj.device)
    loss_spatial_to_semantic = F.cross_entropy(logits, labels)
    loss_semantic_to_spatial = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_spatial_to_semantic + loss_semantic_to_spatial)


def weight_regularization(dynamic_weights: torch.Tensor) -> torch.Tensor:
    """Penalize highly unbalanced multi-objective weight assignments."""
    if dynamic_weights.numel() == 0:
        return torch.tensor(0.0, device=dynamic_weights.device)
    target_sum = torch.ones(dynamic_weights.size(0), device=dynamic_weights.device)
    return F.mse_loss(dynamic_weights.sum(dim=-1), target_sum)


def regression_metrics(prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute RMSE, MAE, and R² using torch-only operations."""
    prediction = prediction.detach()
    target = target.detach()

    mse = F.mse_loss(prediction, target)
    rmse = torch.sqrt(mse)
    mae = F.l1_loss(prediction, target)

    target_mean = target.mean()
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - prediction) ** 2)
    if ss_tot.abs().item() < 1e-12:
        r2 = torch.tensor(0.0, device=target.device)
    else:
        r2 = 1.0 - ss_res / ss_tot

    return {
        "rmse": float(rmse.item()),
        "mae": float(mae.item()),
        "r2": float(r2.item()),
    }
