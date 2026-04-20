"""
Data utilities for the LDTO research prototype.

Expected region-level JSONL format
----------------------------------
{
  "region_id": "wuhan_001",
  "domain": 0,
  "planning_group": "wuhan_core",
  "spatial_features": [0.1, 0.2, ...],
  "semantic_features": [0.01, 0.12, ...],
  "label": 0.73,
  "coverage_gain": 15.2,
  "cost": 4.8,
  "fairness_penalty": 0.19
}

Expected episode-level JSONL format
-----------------------------------
{
  "episode_id": "wuhan_plan_01",
  "budget": 5,
  "candidate_features": [
    [coverage_gain, cost, fairness_penalty, suitability_score],
    ...
  ]
}
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


def load_jsonl(path: str | Path) -> List[dict]:
    """Load a JSONL file into memory."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"No valid records were found in {path}.")
    return records


def infer_feature_dims(records: Sequence[dict]) -> Tuple[int, int]:
    """Infer spatial and semantic dimensions from the first record."""
    first = records[0]
    spatial_dim = len(first["spatial_features"])
    semantic_dim = len(first["semantic_features"])
    return spatial_dim, semantic_dim


def split_records(
    records: Sequence[dict],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Randomly split records into train/validation/test sets."""
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError("Split ratios must sum to 1.0.")
    records = list(records)
    rng = random.Random(seed)
    rng.shuffle(records)
    n = len(records)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = records[:n_train]
    val = records[n_train:n_train + n_val]
    test = records[n_train + n_val:]
    return train, val, test


@dataclass
class RegionTensorBatch:
    """Typed batch container for region-level training."""

    spatial: torch.Tensor
    semantic: torch.Tensor
    label: torch.Tensor
    domain: torch.Tensor


class RegionDataset(Dataset):
    """Dataset for semantic-spatial suitability prediction."""

    def __init__(self, records: Sequence[dict]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        row = self.records[index]
        return {
            "region_id": row.get("region_id", f"region_{index}"),
            "domain": int(row.get("domain", 0)),
            "planning_group": row.get("planning_group", "default_group"),
            "spatial_features": torch.tensor(row["spatial_features"], dtype=torch.float32),
            "semantic_features": torch.tensor(row["semantic_features"], dtype=torch.float32),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "coverage_gain": float(row.get("coverage_gain", 0.0)),
            "cost": float(row.get("cost", 0.0)),
            "fairness_penalty": float(row.get("fairness_penalty", 0.0)),
        }


def collate_region_batch(batch: Sequence[dict]) -> RegionTensorBatch:
    """Collate region-level samples into tensors."""
    spatial = torch.stack([item["spatial_features"] for item in batch], dim=0)
    semantic = torch.stack([item["semantic_features"] for item in batch], dim=0)
    label = torch.stack([item["label"] for item in batch], dim=0)
    domain = torch.tensor([item["domain"] for item in batch], dtype=torch.long)
    return RegionTensorBatch(spatial=spatial, semantic=semantic, label=label, domain=domain)


class PlanningEpisodeDataset(Dataset):
    """Dataset for the dynamic multi-objective RL stage."""

    def __init__(self, episodes: Sequence[dict]):
        self.episodes = list(episodes)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> dict:
        episode = self.episodes[index]
        candidates = torch.tensor(episode["candidate_features"], dtype=torch.float32)
        return {
            "episode_id": episode.get("episode_id", f"episode_{index}"),
            "budget": int(episode.get("budget", 5)),
            "candidate_features": candidates,
        }


def collate_episode_batch(batch: Sequence[dict]) -> dict:
    """Pad candidate features so a mini-batch of episodes can be processed together."""
    max_candidates = max(item["candidate_features"].shape[0] for item in batch)
    feature_dim = batch[0]["candidate_features"].shape[1]
    padded = []
    masks = []
    budgets = []
    ids = []
    for item in batch:
        candidates = item["candidate_features"]
        n = candidates.shape[0]
        pad = torch.zeros(max_candidates - n, feature_dim, dtype=candidates.dtype)
        padded.append(torch.cat([candidates, pad], dim=0))
        mask = torch.zeros(max_candidates, dtype=torch.bool)
        mask[:n] = True
        masks.append(mask)
        budgets.append(item["budget"])
        ids.append(item["episode_id"])
    return {
        "episode_id": ids,
        "budget": torch.tensor(budgets, dtype=torch.long),
        "candidate_features": torch.stack(padded, dim=0),
        "valid_mask": torch.stack(masks, dim=0),
    }


def group_regions_into_episodes(
    records: Sequence[dict],
    group_key: str = "planning_group",
    budget_key: str = "budget",
) -> List[dict]:
    """Build RL episodes directly from region-level records.

    Each region becomes one candidate site with four features:
    [coverage_gain, cost, fairness_penalty, label]
    The label is used as a suitability prior before stage-2 fine-tuning.
    """
    buckets: Dict[str, List[dict]] = {}
    for row in records:
        group_name = str(row.get(group_key, "default_group"))
        buckets.setdefault(group_name, []).append(row)

    episodes: List[dict] = []
    for group_name, rows in buckets.items():
        candidate_features = []
        default_budget = max(1, math.ceil(len(rows) * 0.15))
        for row in rows:
            candidate_features.append([
                float(row.get("coverage_gain", 0.0)),
                float(row.get("cost", 0.0)),
                float(row.get("fairness_penalty", 0.0)),
                float(row.get("label", 0.0)),
            ])
        episodes.append({
            "episode_id": group_name,
            "budget": int(rows[0].get(budget_key, default_budget)),
            "candidate_features": candidate_features,
        })
    return episodes
