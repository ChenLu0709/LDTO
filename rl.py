"""
Dynamic multi-objective reinforcement learning utilities.

Candidate feature convention
----------------------------
The first three dimensions of each candidate are expected to be:
[coverage_gain, cost, fairness_penalty, ...]

Any remaining dimensions are treated as auxiliary context features such as
stage-1 suitability scores or handcrafted planning descriptors.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicWeightPolicy(nn.Module):
    """Policy network that predicts both action logits and objective weights."""

    def __init__(self, candidate_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.action_head = nn.Linear(hidden_dim * 2, 1)
        self.weight_head = nn.Linear(hidden_dim, 3)

    def forward(
        self,
        candidate_features: torch.Tensor,
        selected_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return action logits and dynamic objective weights.

        Args:
            candidate_features: [B, N, D]
            selected_mask: [B, N] with True for already selected sites
            valid_mask: [B, N] with True for real candidates and False for padded slots
        """
        encoded = self.candidate_encoder(candidate_features)  # [B, N, H]
        selected_float = selected_mask.unsqueeze(-1).float()
        remaining_float = (~selected_mask & valid_mask).unsqueeze(-1).float()

        selected_sum = (encoded * selected_float).sum(dim=1)
        remaining_sum = (encoded * remaining_float).sum(dim=1)

        selected_count = selected_float.sum(dim=1).clamp_min(1.0)
        remaining_count = remaining_float.sum(dim=1).clamp_min(1.0)

        selected_mean = selected_sum / selected_count
        remaining_mean = remaining_sum / remaining_count

        state = self.state_encoder(torch.cat([selected_mean, remaining_mean], dim=-1))
        repeated_state = state.unsqueeze(1).expand(-1, encoded.size(1), -1)
        logits = self.action_head(torch.cat([encoded, repeated_state], dim=-1)).squeeze(-1)

        invalid_action_mask = selected_mask | (~valid_mask)
        logits = logits.masked_fill(invalid_action_mask, float("-inf"))

        dynamic_weights = torch.softmax(self.weight_head(state), dim=-1)
        return logits, dynamic_weights


def compute_reward_components(chosen_candidates: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Read the first three candidate dimensions as reward components."""
    return {
        "coverage": chosen_candidates[..., 0],
        "cost": chosen_candidates[..., 1],
        "fairness": chosen_candidates[..., 2],
    }


def reward_from_components(
    components: Dict[str, torch.Tensor],
    dynamic_weights: torch.Tensor,
) -> torch.Tensor:
    """Compute weighted rewards from the three objectives."""
    reward = (
        dynamic_weights[:, 0] * components["coverage"]
        - dynamic_weights[:, 1] * components["cost"]
        - dynamic_weights[:, 2] * components["fairness"]
    )
    return reward


def discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """Compute discounted returns for a sequence of rewards."""
    returns = torch.zeros_like(rewards)
    running = torch.zeros(rewards.size(1), device=rewards.device)
    for t in reversed(range(rewards.size(0))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def run_policy_gradient_episode(
    policy: DynamicWeightPolicy,
    candidate_features: torch.Tensor,
    valid_mask: torch.Tensor,
    budget: torch.Tensor,
    gamma: float = 0.99,
    entropy_coef: float = 1e-3,
) -> Dict[str, torch.Tensor]:
    """Run one batched planning episode and return the REINFORCE objective."""
    device = candidate_features.device
    batch_size, n_candidates, _ = candidate_features.shape
    max_steps = int(budget.max().item())

    selected_mask = torch.zeros(batch_size, n_candidates, dtype=torch.bool, device=device)

    log_probs = []
    rewards = []
    entropies = []
    dynamic_weight_history = []

    for step in range(max_steps):
        logits, dynamic_weights = policy(candidate_features, selected_mask, valid_mask)
        action_distribution = torch.distributions.Categorical(logits=logits)
        actions = action_distribution.sample()

        chosen_candidates = candidate_features[torch.arange(batch_size, device=device), actions]
        components = compute_reward_components(chosen_candidates)
        reward = reward_from_components(components, dynamic_weights)

        budget_mask = (step < budget).float()
        reward = reward * budget_mask

        log_prob = action_distribution.log_prob(actions) * budget_mask
        entropy = action_distribution.entropy() * budget_mask

        selected_mask[torch.arange(batch_size, device=device), actions] = True

        log_probs.append(log_prob)
        rewards.append(reward)
        entropies.append(entropy)
        dynamic_weight_history.append(dynamic_weights)

    reward_tensor = torch.stack(rewards, dim=0)
    log_prob_tensor = torch.stack(log_probs, dim=0)
    entropy_tensor = torch.stack(entropies, dim=0)
    dynamic_weight_tensor = torch.stack(dynamic_weight_history, dim=0)

    returns = discounted_returns(reward_tensor, gamma=gamma)
    baseline = returns.mean(dim=1, keepdim=True)
    advantages = returns - baseline

    policy_loss = -(log_prob_tensor * advantages.detach()).mean()
    entropy_bonus = entropy_tensor.mean()
    mean_dynamic_weights = dynamic_weight_tensor.mean(dim=0)

    return {
        "loss": policy_loss - entropy_coef * entropy_bonus,
        "episode_reward": reward_tensor.sum(dim=0).mean(),
        "dynamic_weights": mean_dynamic_weights,
        "selected_mask": selected_mask,
    }
