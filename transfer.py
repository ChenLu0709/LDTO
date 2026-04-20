"""
Cross-region transfer learning utilities.

This file contains:
1. Gradient reversal for domain-adversarial learning.
2. A lightweight domain discriminator.
3. A one-step meta-transfer objective for source-to-target adaptation.
"""

from __future__ import annotations

import copy
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFunction(torch.autograd.Function):
    """Autograd trick that multiplies incoming gradients by -lambda."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_grl: float) -> torch.Tensor:
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_grl * grad_output, None


def gradient_reverse(x: torch.Tensor, lambda_grl: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal to a tensor."""
    return GradientReversalFunction.apply(x, lambda_grl)


class DomainDiscriminator(nn.Module):
    """Binary discriminator that predicts whether a sample is source or target."""

    def __init__(self, feature_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def domain_adversarial_loss(
    fused_features: torch.Tensor,
    domains: torch.Tensor,
    discriminator: DomainDiscriminator,
    lambda_grl: float = 1.0,
) -> torch.Tensor:
    """Compute the domain-adversarial loss with a gradient reversal layer."""
    unique_domains = torch.unique(domains)
    if unique_domains.numel() < 2:
        return torch.tensor(0.0, device=fused_features.device)
    reversed_features = gradient_reverse(fused_features, lambda_grl=lambda_grl)
    logits = discriminator(reversed_features)
    return F.cross_entropy(logits, domains)


def _task_loss_from_batch(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Internal helper used by meta_transfer_loss."""
    outputs = model(batch["spatial"], batch["semantic"])
    return F.mse_loss(outputs["score"], batch["label"])


def meta_transfer_loss(
    model: nn.Module,
    source_batch: Dict[str, torch.Tensor],
    target_batch: Dict[str, torch.Tensor],
    inner_lr: float = 5e-5,
) -> torch.Tensor:
    """Approximate the two-stage source-to-target meta adaptation.

    The implementation is intentionally lightweight and dependency-free:
    a temporary copy of the model is updated with a one-step source loss,
    and the adapted copy is evaluated on the target batch.
    """
    if source_batch["label"].numel() == 0 or target_batch["label"].numel() == 0:
        device = source_batch["label"].device if source_batch["label"].numel() > 0 else target_batch["label"].device
        return torch.tensor(0.0, device=device)

    adapted_model = copy.deepcopy(model)
    adapted_model.train()

    source_loss = _task_loss_from_batch(adapted_model, source_batch)
    grads = torch.autograd.grad(source_loss, adapted_model.parameters(), allow_unused=True)

    with torch.no_grad():
        for parameter, grad in zip(adapted_model.parameters(), grads):
            if grad is not None:
                parameter -= inner_lr * grad

    target_loss = _task_loss_from_batch(adapted_model, target_batch)
    return target_loss


def split_source_target_batch(batch: Dict[str, torch.Tensor]) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Split a mixed mini-batch into source and target sub-batches."""
    domains = batch["domain"]
    source_mask = domains == 0
    target_mask = domains == 1

    def _slice(mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "spatial": batch["spatial"][mask],
            "semantic": batch["semantic"][mask],
            "label": batch["label"][mask],
        }

    return _slice(source_mask), _slice(target_mask)
