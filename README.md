# 🧠🗺️ LDTO: LLM-Enhanced Deep Transfer Optimization for Public Sports Facility Siting

> A clean **research-oriented Python prototype** that implements the core ideas of  
> **Semantic–Spatial Collaborative Alignment (SSCA)**,  
> **Cross-Region Transfer Adaptation (CRTA)**, and  
> **Dynamic Multi-Objective Reinforcement Learning (DMRL)**.

<div align="center">

**Semantic Understanding × Spatial Representation × Transfer Generalization × Policy Optimization**

</div>

---

## ✨ Overview

This repository provides a compact yet academically structured implementation of the **LDTO** framework for **public sports facility location planning**.  
The codebase is organized to reflect the methodological logic of the paper:

1. **Policy semantics** are encoded as dense vectors.
2. **Spatial-demographic features** are learned through a dedicated branch.
3. A **semantic–spatial alignment module** fuses both modalities.
4. A **cross-region transfer mechanism** improves robustness under domain shift.
5. A **dynamic multi-objective RL policy** performs adaptive site-selection optimization.

The project is intentionally designed as a **research prototype** rather than a production system, which makes it easier to extend for ablation studies, new datasets, and reviewer-oriented reproducibility.

---

## 🏛️ Methodological Mapping

| Paper Module | Code Realization | Main File |
|---|---|---|
| Semantic–Spatial Collaborative Alignment | Dual encoders + symmetric InfoNCE alignment | `models.py`, `losses.py` |
| Cross-Region Transfer Adaptation | Gradient reversal + domain discriminator + one-step meta adaptation | `transfer.py` |
| Dynamic Multi-Objective Reinforcement Learning | Policy network with dynamic objective weights | `rl.py` |
| Unified Optimization | Stage-1 supervised learning + optional Stage-2 RL fine-tuning | `train.py` |

---

## 📁 Project Structure

```text
LDTO_codebase/
├── config.py      # Centralized experiment and model configuration
├── data.py        # JSONL loading, dataset wrappers, collators, episode builders
├── models.py      # Semantic encoder, spatial encoder, fusion module, predictor
├── losses.py      # Prediction, alignment, regularization, and metrics
├── transfer.py    # Domain adversarial learning and meta-transfer utilities
├── rl.py          # Dynamic objective weighting and policy-gradient optimization
├── train.py       # Full training entry point
└── README.md      # Project documentation
```

---

## 🔬 Core Design Principles

### 1. Semantic–Spatial Dual-Branch Learning
The model explicitly separates:
- **Spatial structure learning** from demographic and geographic variables.
- **Policy semantic learning** from precomputed semantic embeddings.

This separation is useful because it preserves the heterogeneity of the two modalities while enabling interpretable fusion downstream.

### 2. Contrastive Cross-Modal Alignment
The **InfoNCE-based alignment loss** encourages one-to-one correspondence between spatial and semantic representations in latent space, improving policy consistency in prediction.

### 3. Cross-Region Adaptation
The transfer module combines:
- **Domain-adversarial learning** for distribution alignment.
- **One-step meta adaptation** for source-to-target parameter transfer.

This design is particularly suitable for **data-scarce urban regions**.

### 4. Dynamic Multi-Objective Optimization
Instead of using fixed weights, the RL policy learns dynamic weights for:
- **Coverage gain**
- **Cost penalty**
- **Fairness penalty**

This allows adaptive balancing across different planning contexts.

---

## ⚙️ Environment

Recommended environment:

- **Python** 3.10+
- **PyTorch** 2.1+
- **CUDA** optional but recommended for large-scale experiments

Minimal installation:

```bash
pip install torch
```

Optional scientific stack:

```bash
pip install numpy pandas scikit-learn matplotlib
```

---

## 📦 Input Data Format

### Region-Level Data (`regions.jsonl`)

Each line should be a JSON object:

```json
{
  "region_id": "wuhan_001",
  "domain": 0,
  "planning_group": "wuhan_core",
  "spatial_features": [0.12, 0.44, 0.31, 0.09],
  "semantic_features": [0.001, 0.032, 0.114, 0.228],
  "label": 0.73,
  "coverage_gain": 15.2,
  "cost": 4.8,
  "fairness_penalty": 0.19
}
```

### Episode-Level Data (`episodes.jsonl`, optional)

```json
{
  "episode_id": "wuhan_plan_01",
  "budget": 5,
  "candidate_features": [
    [15.2, 4.8, 0.19, 0.73],
    [11.4, 3.9, 0.27, 0.68]
  ]
}
```

> If `episodes.jsonl` is not provided, the code can automatically group region-level records into RL episodes using `planning_group`.

## 🧪 Case Study Configuration (Wuhan)

This repository includes a **Wuhan-oriented case-study setting** for demonstrating how LDTO improves facility allocation under regional heterogeneity, policy constraints, and data scarcity. The case study corresponds to the manuscript discussion of Figure 4 and is intended to make the experimental setting more transparent and reproducible.

### 1. Study Area and Spatial Unit

- **Study area:** Wuhan, China.
- **Spatial unit:** street-level regions.
- **Sample size:** **168 regional samples**.
- **Regional grouping:**
  - **Source domain:** central urban areas with relatively complete facility-layout observations and denser socioeconomic records.
  - **Target domain:** suburban or newly developed peripheral areas with relatively limited samples and weaker historical layout density.

This source–target split is used to evaluate whether the model can transfer planning knowledge from data-rich urban cores to data-scarce outer districts while maintaining policy consistency and spatial equity.

### 2. Data Types Used in the Case Study

The case study integrates **three primary data modalities** plus **derived optimization indicators**:

#### (a) Geospatial features
Used to describe the physical and locational characteristics of each street-level unit, including but not limited to:
- land-use structure,
- road-network density,
- transportation accessibility,
- elevation or terrain-related indicators,
- candidate-site spatial position.

#### (b) Demographic and socioeconomic features
Used to characterize service demand and regional development intensity, such as:
- population density,
- residential concentration,
- development intensity,
- socioeconomic conditions,
- potential public-service demand.

#### (c) Policy text data
Each region is associated with local planning or governance text used to represent policy intent. In the manuscript setting:
- each region is linked with approximately **1,000 words** of policy-related text,
- the text is encoded into **768-dimensional semantic vectors**,
- semantic vectors are then fused with structured spatial features through the **SSCA** module.

#### (d) Derived optimization targets / decision variables
To support prediction and RL-based planning, the following variables are used:
- **historical comprehensive benefit score** (`label`),
- **coverage gain**,
- **construction / maintenance cost**,
- **fairness penalty**,
- region/domain/group identifiers for transfer learning and episode construction.

### 3. Data Split and Data Ratio

For the supervised prediction stage, the case-study dataset follows the same split as described in the manuscript:

- **Training set:** 70%
- **Validation set:** 20%
- **Test set:** 10%

In practice, the following rules should be respected:
- the split should be **deterministic under a fixed random seed**,
- the **source-domain / target-domain distinction must be preserved**,
- no region may appear in more than one subset,
- normalization statistics should be fitted on the **training set only** and then applied to validation/test data.

For transfer experiments, it is recommended that the **target-domain training subset remain smaller than the source-domain subset**, so that the benefit of CRTA can be meaningfully evaluated under realistic data-scarce conditions.

### 4. Data Augmentation Setting for the Case Study

The manuscript compares facility distributions **before** and **after data augmentation**. To make this process reproducible in the repository, the case study should document augmentation at the **training-data level**, rather than modifying the evaluation labels or directly editing the final map output.

A practical augmentation protocol consistent with the LDTO framework is as follows:

#### (a) Augmentation objective
The goal of augmentation is to alleviate the scarcity and distribution instability of peripheral regions, especially in the **target domain**, so that the model can learn more robust suitability patterns before RL-based layout optimization.

#### (b) Recommended augmentation scope
- apply augmentation to the **training subset only**,
- prioritize **target-domain** samples and underrepresented planning groups,
- do **not** augment validation or test sets,
- keep the original, non-augmented records for final evaluation and visualization comparison.

#### (c) Recommended augmentation forms
The repository can support one or more of the following feature-level strategies:

1. **Numerical-feature perturbation**  
   Apply small stochastic perturbations to normalized geospatial and socioeconomic variables to simulate mild observational variation while preserving the original regional pattern.

2. **Intra-domain interpolation / mixup-style synthesis**  
   Construct additional samples by interpolating between nearby or same-domain regions with similar planning characteristics. This is especially useful when suburban samples are sparse.

3. **Semantic-preserving augmentation**  
   Keep the original policy meaning unchanged while reusing the same semantic vector or a semantically equivalent embedding representation. The purpose is to prevent semantic drift during augmentation.

4. **Domain-balancing oversampling**  
   Increase the effective number of target-domain training records so that the source/target learning process becomes less biased toward the urban core.

#### (d) Augmentation ratio
Because the manuscript emphasizes **data scarcity in peripheral regions**, a practical default is:
- keep the **raw source-domain data unchanged or only lightly augmented**,
- augment the **target-domain training subset** until its effective sample size is closer to that of the source domain,
- use a **moderate augmentation ratio** to avoid overwhelming the original data distribution.

In reproducibility-oriented experiments, a common setting is to move the target-domain training subset toward a roughly balanced condition with the source domain, while still preserving the fact that it is the harder domain.

#### (e) Important augmentation constraints
- augmented samples must remain within a **reasonable feature range** after normalization or inverse transformation,
- no augmented sample should leak information from validation/test subsets,
- semantic embeddings should remain aligned with the planning intent of the original region,
- augmentation must not alter the original ground-truth labels in validation/test evaluation,
- augmentation should be logged for reproducibility.

### 5. Optimization Constraints in the Case Study

The case study is not a free-form ranking task; it is a constrained planning problem. The following constraints should be explicitly documented in the README.

#### (a) Candidate-region constraint
Only predefined street-level candidate regions are eligible for facility placement. The policy network selects from the remaining candidate set at each decision step.

#### (b) No-duplicate selection constraint
Once a candidate region has been selected in the current planning episode, it should not be selected again.

#### (c) Budget / quota constraint
The episode should include a **budget**, **maximum number of selected sites**, or equivalent planning quota. The RL episode terminates when the budget is exhausted or the planning horizon is reached.

#### (d) Multi-objective constraint
Optimization must jointly consider:
- maximizing **service coverage gain**,
- minimizing **construction cost**,
- reducing **regional fairness penalty**.

This is implemented as a dynamic objective-balancing process rather than a fixed-weight linear rule.

#### (e) Policy-consistency constraint
Site selection should remain consistent with policy semantics extracted from planning texts. In other words, candidate sites are not chosen solely by geometry or demand density, but also by whether they align with planning priorities such as universal fitness and regional balance.

#### (f) Cross-region transfer constraint
The transfer-learning stage should preserve the distinction between source and target domains. Adaptation is performed through adversarial alignment and meta-updating rather than by collapsing all regions into one homogeneous dataset.

### 6. Suggested JSON Fields for Reproducible Case-Study Use

For a more explicit case-study configuration, each region-level record may include:

```json
{
  "region_id": "wuhan_001",
  "domain": 0,
  "planning_group": "wuhan_core",
  "spatial_features": [0.12, 0.44, 0.31, 0.09],
  "semantic_features": [0.001, 0.032, 0.114, 0.228],
  "label": 0.73,
  "coverage_gain": 15.2,
  "cost": 4.8,
  "fairness_penalty": 0.19,
  "is_augmented": false,
  "augmentation_type": "none"
}
```

If an augmented sample is generated for the target domain, the metadata should be updated accordingly, for example:
- `is_augmented: true`
- `augmentation_type: "feature_noise"`, `"interpolation"`, or `"oversampling"`
- `source_region_id` or `parent_region_ids` for traceability

### 7. How Figure 4 Should Be Interpreted

In the case-study narrative, the **left panel** represents the original facility distribution or the raw planning baseline, while the **right panel** represents the optimized outcome after applying the LDTO pipeline under the above setting. The improvement should therefore be interpreted as the combined result of:

1. semantic–spatial fusion,
2. cross-region knowledge transfer,
3. data augmentation for sparse regions,
4. and dynamic multi-objective planning optimization.

### 8. Reproducibility Recommendation

To make the case study easier for reviewers and readers to reproduce, it is recommended to provide:
- the raw and augmented sample counts for each domain,
- the exact augmentation operators used,
- the number of candidate regions per planning episode,
- the budget or site quota,
- and the seed used for data splitting and training.

If these values are changed for a new city or a new experimental protocol, they should be reported together with the corresponding visualization results.

---

---

## 🚀 Quick Start

### Stage 1: Train the prediction backbone

```bash
python train.py \
    --regions data/regions.jsonl \
    --output_dir outputs
```

### Stage 2: Train the dynamic RL planner

```bash
python train.py \
    --regions data/regions.jsonl \
    --episodes data/episodes.jsonl \
    --run_rl \
    --output_dir outputs
```

---

## 🧪 Reproducibility Notes

To keep experiments reproducible:

- Random seeds are fixed in `train.py`.
- Train/validation/test splitting is deterministic under a given seed.
- Checkpoints and metric files are automatically saved to the output directory.
- The implementation avoids unnecessary external dependencies.

Generated outputs include:

- `ldto_best.pt`
- `prediction_metrics.json`
- `ldto_policy_best.pt` (if RL is enabled)
- `rl_metrics.json` (if RL is enabled)
- `summary.json`

---

## 📈 Suggested Extensions for Journal / Conference Submission

This prototype is deliberately modular so that it can be extended into a submission-ready experimental codebase. Recommended directions include:

- **Ablation settings**
  - Remove SSCA / CRTA / DMRL independently
  - Replace the semantic backbone with alternative embedding generators
- **Stronger transfer protocols**
  - Multi-source adaptation
  - Region-level curriculum transfer
- **Richer RL environments**
  - Explicit budget constraints
  - Spatial adjacency graphs
  - Coverage simulation based on road-network accessibility
- **Interpretability**
  - Visualize semantic–spatial latent alignment
  - Report dynamic objective weights over planning stages

---

## 🧭 Practical Notes

- The current implementation assumes that **policy text embeddings are already prepared**.
- If the original pipeline uses GPT-based semantic parsing, embeddings can be generated offline and stored as `semantic_features`.
- The RL module expects the **first three candidate dimensions** to correspond to:
  1. `coverage_gain`
  2. `cost`
  3. `fairness_penalty`

---

## 📝 Citation-Style Project Description

If you use this code structure in an academic manuscript, a concise description can be:

> We implemented the proposed LDTO framework in Python/PyTorch, following a two-stage optimization pipeline consisting of semantic–spatial supervised learning and dynamic multi-objective reinforcement learning. The implementation includes explicit semantic–spatial contrastive alignment, domain-adversarial transfer learning, and policy-gradient-based dynamic objective balancing.

---

## 🤝 Final Remark

This repository is designed to be **clean, extendable, and academically presentable**.  
It should serve well as a starting point for:

- journal submission supplements,
- conference reproducibility packages,
- ablation experiments,
- and future methodological expansion.

---

<div align="center">

**LDTO Research Prototype**  
*Elegant enough for presentation, modular enough for experimentation.*

</div>
