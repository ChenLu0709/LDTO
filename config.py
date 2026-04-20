"""
Configuration definitions for the LDTO research prototype.

All comments are intentionally written in English so the project can be
used directly in an international academic context.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ModelConfig:
    """Hyperparameters for the semantic-spatial prediction model."""

    spatial_dim: int
    semantic_dim: int = 768
    hidden_dim: int = 256
    projection_dim: int = 128
    fusion_dim: int = 256
    dropout: float = 0.30


@dataclass
class LossWeights:
    """Relative weights for different objectives in the unified loss."""

    prediction: float = 1.0
    alignment: float = 0.2
    adversarial: float = 0.1
    meta_transfer: float = 0.1
    rl_regularization: float = 1e-3


@dataclass
class TrainingConfig:
    """Training configuration for the two-stage pipeline."""

    seed: int = 42
    batch_size: int = 32
    epochs: int = 120
    rl_epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    inner_lr: float = 5e-5
    dropout: float = 0.30
    temperature: float = 0.07
    lambda_grl: float = 1.0
    early_stopping_patience: int = 20
    gamma: float = 0.99
    entropy_coef: float = 1e-3
    max_grad_norm: float = 5.0
    train_ratio: float = 0.70
    val_ratio: float = 0.20
    test_ratio: float = 0.10

    def split_ratios(self) -> Tuple[float, float, float]:
        """Return split ratios and validate that they sum to one."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-8:
            raise ValueError("Train/val/test ratios must sum to 1.0.")
        return self.train_ratio, self.val_ratio, self.test_ratio


@dataclass
class RLConfig:
    """Configuration for the dynamic multi-objective policy network."""

    candidate_dim: int = 4
    hidden_dim: int = 128
    budget_default: int = 5
    gamma: float = 0.99
    entropy_coef: float = 1e-3
    regularization_coef: float = 1e-3
    initial_weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)


@dataclass
class Paths:
    """Convenience container for file system paths."""

    train_jsonl: str = "data/regions.jsonl"
    episode_jsonl: str = "data/episodes.jsonl"
    output_dir: str = "outputs"
    checkpoint_name: str = "ldto_best.pt"


DEFAULT_LOSS_WEIGHTS = LossWeights()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_RL_CONFIG = RLConfig()
