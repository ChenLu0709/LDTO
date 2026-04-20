"""
Training entry point for the LDTO research prototype.

The script supports:
1. Region-level prediction with semantic-spatial alignment and transfer adaptation.
2. Optional RL fine-tuning for dynamic multi-objective site selection.

Example
-------
python train.py \
    --regions data/regions.jsonl \
    --episodes data/episodes.jsonl \
    --output_dir outputs
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from config import DEFAULT_LOSS_WEIGHTS, DEFAULT_RL_CONFIG, DEFAULT_TRAINING_CONFIG, ModelConfig
from data import (
    PlanningEpisodeDataset,
    RegionDataset,
    collate_episode_batch,
    collate_region_batch,
    group_regions_into_episodes,
    infer_feature_dims,
    load_jsonl,
    split_records,
)
from losses import contrastive_alignment_loss, prediction_loss, regression_metrics, weight_regularization
from models import LDTOModel
from rl import DynamicWeightPolicy, run_policy_gradient_episode
from transfer import DomainDiscriminator, domain_adversarial_loss, meta_transfer_loss, split_source_target_batch


def set_seed(seed: int) -> None:
    """Make the training pipeline as reproducible as possible."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batch_to_device(batch, device: torch.device):
    """Move a dataclass-like or dict batch to the target device."""
    if hasattr(batch, "__dict__"):
        for key, value in batch.__dict__.items():
            if torch.is_tensor(value):
                setattr(batch, key, value.to(device))
        return batch
    for key, value in batch.items():
        if torch.is_tensor(value):
            batch[key] = value.to(device)
    return batch


def evaluate(
    model: LDTOModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate the prediction model on a validation or test split."""
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            outputs = model(batch.spatial, batch.semantic)
            all_predictions.append(outputs["score"])
            all_targets.append(batch.label)

    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    return regression_metrics(predictions, targets)


def train_prediction_model(args) -> Tuple[LDTOModel, Dict[str, float], list[dict]]:
    """Train the semantic-spatial prediction model."""
    train_cfg = DEFAULT_TRAINING_CONFIG
    loss_weights = DEFAULT_LOSS_WEIGHTS
    set_seed(train_cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    records = load_jsonl(args.regions)
    spatial_dim, semantic_dim = infer_feature_dims(records)
    train_records, val_records, test_records = split_records(
        records=records,
        seed=train_cfg.seed,
        train_ratio=train_cfg.train_ratio,
        val_ratio=train_cfg.val_ratio,
        test_ratio=train_cfg.test_ratio,
    )

    model_cfg = ModelConfig(
        spatial_dim=spatial_dim,
        semantic_dim=semantic_dim,
        hidden_dim=args.hidden_dim,
        projection_dim=args.projection_dim,
        fusion_dim=args.fusion_dim,
        dropout=train_cfg.dropout,
    )

    model = LDTOModel(model_cfg).to(device)
    discriminator = DomainDiscriminator(feature_dim=model_cfg.fusion_dim, hidden_dim=model_cfg.hidden_dim).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(discriminator.parameters()),
        lr=args.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    train_loader = DataLoader(
        RegionDataset(train_records),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_region_batch,
    )
    val_loader = DataLoader(
        RegionDataset(val_records),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_region_batch,
    )
    test_loader = DataLoader(
        RegionDataset(test_records),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_region_batch,
    )

    best_val_rmse = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            batch = batch_to_device(batch, device)
            optimizer.zero_grad()

            outputs = model(batch.spatial, batch.semantic)
            pred_loss = prediction_loss(outputs["score"], batch.label)
            align_loss = contrastive_alignment_loss(
                outputs["spatial_proj"],
                outputs["semantic_proj"],
                temperature=args.temperature,
            )
            adv_loss = domain_adversarial_loss(
                outputs["fused"],
                batch.domain,
                discriminator=discriminator,
                lambda_grl=args.lambda_grl,
            )

            meta_loss = torch.tensor(0.0, device=device)
            mixed_batch = {
                "spatial": batch.spatial,
                "semantic": batch.semantic,
                "label": batch.label,
                "domain": batch.domain,
            }
            source_batch, target_batch = split_source_target_batch(mixed_batch)
            if source_batch["label"].numel() > 0 and target_batch["label"].numel() > 0:
                meta_loss = meta_transfer_loss(model, source_batch, target_batch, inner_lr=args.inner_lr)

            total_loss = (
                loss_weights.prediction * pred_loss
                + loss_weights.alignment * align_loss
                + loss_weights.adversarial * adv_loss
                + loss_weights.meta_transfer * meta_loss
            )

            total_loss.backward()
            clip_grad_norm_(list(model.parameters()) + list(discriminator.parameters()), train_cfg.max_grad_norm)
            optimizer.step()

            epoch_loss += float(total_loss.item())

        val_metrics = evaluate(model, val_loader, device)
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_state = {
                "model": model.state_dict(),
                "discriminator": discriminator.state_dict(),
                "model_config": vars(model_cfg),
                "args": vars(args),
            }
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"[Epoch {epoch + 1:03d}] "
            f"train_loss={epoch_loss / max(len(train_loader), 1):.4f} "
            f"val_rmse={val_metrics['rmse']:.4f} "
            f"val_mae={val_metrics['mae']:.4f} "
            f"val_r2={val_metrics['r2']:.4f}"
        )

        if patience_counter >= train_cfg.early_stopping_patience:
            print("Early stopping was triggered.")
            break

    if best_state is None:
        raise RuntimeError("Training failed before any valid checkpoint was created.")

    model.load_state_dict(best_state["model"])
    test_metrics = evaluate(model, test_loader, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "ldto_best.pt"
    torch.save(best_state, checkpoint_path)

    metrics_path = output_dir / "prediction_metrics.json"
    metrics_path.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    print(f"Saved best prediction checkpoint to: {checkpoint_path}")
    print(f"Saved prediction metrics to: {metrics_path}")
    return model, test_metrics, records


def train_rl_stage(args, records: list[dict], device: torch.device) -> Dict[str, float]:
    """Train the dynamic multi-objective RL policy."""
    rl_cfg = DEFAULT_RL_CONFIG
    loss_weights = DEFAULT_LOSS_WEIGHTS

    if args.episodes:
        episodes = load_jsonl(args.episodes)
    else:
        episodes = group_regions_into_episodes(records)

    dataset = PlanningEpisodeDataset(episodes)
    loader = DataLoader(
        dataset,
        batch_size=args.rl_batch_size,
        shuffle=True,
        collate_fn=collate_episode_batch,
    )

    candidate_dim = dataset[0]["candidate_features"].shape[1]
    policy = DynamicWeightPolicy(candidate_dim=candidate_dim, hidden_dim=args.rl_hidden_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.rl_learning_rate)

    best_reward = float("-inf")
    best_state = None

    for epoch in range(args.rl_epochs):
        policy.train()
        rewards = []

        for batch in loader:
            batch = batch_to_device(batch, device)
            optimizer.zero_grad()

            result = run_policy_gradient_episode(
                policy=policy,
                candidate_features=batch["candidate_features"],
                valid_mask=batch["valid_mask"],
                budget=batch["budget"],
                gamma=args.gamma,
                entropy_coef=args.entropy_coef,
            )

            reg_loss = weight_regularization(result["dynamic_weights"]) * loss_weights.rl_regularization
            total_loss = result["loss"] + reg_loss
            total_loss.backward()
            clip_grad_norm_(policy.parameters(), DEFAULT_TRAINING_CONFIG.max_grad_norm)
            optimizer.step()

            rewards.append(float(result["episode_reward"].item()))

        mean_reward = sum(rewards) / max(len(rewards), 1)
        print(f"[RL Epoch {epoch + 1:03d}] mean_episode_reward={mean_reward:.4f}")

        if mean_reward > best_reward:
            best_reward = mean_reward
            best_state = policy.state_dict()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = output_dir / "ldto_policy_best.pt"
    torch.save(best_state, policy_path)

    metrics = {"mean_episode_reward": best_reward}
    metrics_path = output_dir / "rl_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved best RL policy to: {policy_path}")
    print(f"Saved RL metrics to: {metrics_path}")
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser."""
    parser = argparse.ArgumentParser(description="Train the LDTO research prototype.")
    parser.add_argument("--regions", type=str, required=True, help="Path to region-level JSONL data.")
    parser.add_argument("--episodes", type=str, default="", help="Optional path to episode-level JSONL data.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory for checkpoints and metrics.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training even if CUDA is available.")

    parser.add_argument("--batch_size", type=int, default=DEFAULT_TRAINING_CONFIG.batch_size)
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAINING_CONFIG.epochs)
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_TRAINING_CONFIG.learning_rate)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TRAINING_CONFIG.temperature)
    parser.add_argument("--lambda_grl", type=float, default=DEFAULT_TRAINING_CONFIG.lambda_grl)
    parser.add_argument("--inner_lr", type=float, default=DEFAULT_TRAINING_CONFIG.inner_lr)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--fusion_dim", type=int, default=256)

    parser.add_argument("--run_rl", action="store_true", help="Enable stage-2 RL fine-tuning.")
    parser.add_argument("--rl_epochs", type=int, default=DEFAULT_TRAINING_CONFIG.rl_epochs)
    parser.add_argument("--rl_batch_size", type=int, default=4)
    parser.add_argument("--rl_hidden_dim", type=int, default=128)
    parser.add_argument("--rl_learning_rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=DEFAULT_TRAINING_CONFIG.gamma)
    parser.add_argument("--entropy_coef", type=float, default=DEFAULT_TRAINING_CONFIG.entropy_coef)
    return parser


def main() -> None:
    """Run the full training workflow."""
    parser = build_arg_parser()
    args = parser.parse_args()

    model, prediction_metrics, records = train_prediction_model(args)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if args.run_rl:
        rl_metrics = train_rl_stage(args, records, device)
    else:
        rl_metrics = {"status": "RL stage skipped."}

    summary = {
        "prediction_stage": prediction_metrics,
        "rl_stage": rl_metrics,
    }
    summary_path = Path(args.output_dir) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved run summary to: {summary_path}")


if __name__ == "__main__":
    main()
