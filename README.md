# EleRPO: Element Relative Policy Optimization

This repository contains the minimal implementation of **EleRPO**, an extension of Group Relative Policy Optimization (GRPO) that provides fine-grained, per-token advantage signals for structured generation tasks such as object detection.

## Algorithm

### Overview

The hybrid advantage for token *t* in rollout *i* is:

```
A(i, t) = 0.5 * A_seq(i) + 0.5 * A_box(i, t)
```

where `A_seq` is the standard GRPO/GDPO sequence-level advantage (one scalar per rollout) and `A_box` is the per-token element-level advantage (varies across tokens).

### Element-Level Advantage: Delta-F1 with Prefix Conditioning

For a rollout that produces boxes `y_1, y_2, ..., y_n`, the reward for box *k* is the incremental F1 gain:

```
ΔF1_k = F1(y_{1..k}, GT) − F1(y_{1..k−1}, GT)
```

where F1 is the type-separated weighted Hungarian IoU F1:

```
F1 = w_vis * F1_hungarian(GT_visual, pred_visual) + w_txt * F1_hungarian(GT_text, pred_text)
```

**Prefix conditioning**: Rollouts at the same position *k* may have different prefix contexts. A rollout that has already matched all GT boxes will have ΔF1 ≈ 0 regardless of box quality, while one with poor prefix coverage may show large ΔF1 gains. To account for this, we cluster rollouts by their prefix F1 value `F1(y_{1..k−1})` and subtract the cluster-conditional mean:

```
r̂_k^(i) = ΔF1_k^(i) − mean_{j ∈ C(i)} ΔF1_k^(j)
```

where `C(i)` is the cluster containing rollout *i*, determined by 1-D k-means on prefix F1 values.

**Silhouette-based adaptive clustering**: Instead of fixing the number of clusters, we try `k = 2, 3, ..., K_max` and select the *k* with the highest mean silhouette score. If the best score is below a threshold (default 0.5), the data has no clear cluster structure and we fall back to global mean subtraction (equivalent to standard delta-F1).

**Scale normalization**: Finally, per-position RMS normalization is applied:

```
A_box(i, k) = r̂_k^(i) / sqrt(mean_j(r̂_k^(j)²) + ε)
```

### Sequence-Level Advantage: GDPO

For the sequence-level component, we use GDPO (Group reward-Decoupled Policy Optimization), which normalizes each reward function separately within generation groups before combining:

1. For each reward function *f*, z-score within the generation group.
2. Weighted sum: `A = Σ_f w_f * A_f`.
3. Batch-wise z-score for stable magnitude.

### Pipeline

```
Model Rollout
    │
    ├──> Sequence-Level Rewards ──> GDPO Advantage ──> A_seq (scalar per rollout)
    │
    └──> Parse into boxes ──> Hungarian Matching with GT
              │
              └──> Prefix F1 computation ──> ΔF1 per box
                        │
                        └──> Silhouette clustering on prefix F1
                                  │
                                  └──> Cluster-conditional mean subtraction
                                            │
                                            └──> RMS normalization ──> A_box (per-token)

    Final: A(t) = 0.5 * A_seq + 0.5 * A_box(t)  →  GRPO policy loss
```

## Repository Structure

```
elerpo/
├── README.md
├── stepwise/                      # Core stepwise module (self-contained)
│   ├── __init__.py
│   ├── iou_utils.py               # IoU matrix, Hungarian matching, F1 reward
│   ├── segment.py                 # Segment / SlicingResult data structures
│   ├── slicing_strategy.py        # HungarianSlicing: parse + match
│   └── segment_reward.py          # SegmentIoUF1Reward + delta_f1_cond_prefix
├── integration/                   # How to integrate into ms-swift
│   ├── gdpo.py                    # GDPO advantage estimator
│   ├── grpo_config_fields.py      # New GRPOConfig fields to add
│   └── grpo_trainer_patch.py      # Exact trainer modifications
└── examples/
    └── config_example.yaml        # Example training config
```

## Integration into ms-swift

### Prerequisites

- [ms-swift](https://github.com/modelscope/ms-swift) >= 3.x
- [TRL](https://github.com/huggingface/trl) (used by ms-swift for `GRPOTrainer`)
- numpy, scipy

### Step 1: Install the stepwise package

Copy the `stepwise/` directory into your project, or install it as a package:

```bash
pip install -e .   # from the elerpo repo root
```

### Step 2: Add config fields

Add the fields from [`integration/grpo_config_fields.py`](integration/grpo_config_fields.py) to your `GRPOConfig` dataclass in `swift/trainers/arguments.py`.

### Step 3: Modify the trainer

Apply the changes described in [`integration/grpo_trainer_patch.py`](integration/grpo_trainer_patch.py) to `swift/trainers/rlhf_trainer/grpo_trainer.py`. There are four integration points:

1. **Import** the stepwise module (top of file).
2. **Initialize** stepwise components (end of `__init__`).
3. **Route** to hybrid scoring (in `_generate_and_score_completions`).
4. **Handle 2D advantages** in the loss function.

### Step 4: Add GDPO (optional)

If using GDPO as the sequence-level advantage estimator, add the code from [`integration/gdpo.py`](integration/gdpo.py) and wire it into `_compute_advantages`.

### Step 5: Configure training

See [`examples/config_example.yaml`](examples/config_example.yaml) for an example configuration.

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stepwise_training` | `false` | Enable stepwise per-token advantages |
| `stepwise_norm_mode` | `batch` | `delta_f1_cond_prefix` recommended |
| `stepwise_n_clusters` | 2 | Max clusters for silhouette clustering |
| `stepwise_min_silhouette` | 0.5 | Min score to apply clustering |
| `stepwise_hybrid_mode` | `average` | `0.5 * seq + 0.5 * box` |
| `stepwise_delta_f1_weights` | None | Per-type F1 weights, e.g. `[0.5, 0.5]` |
| `stepwise_check_labels` | `true` | Require type match for reward |
| `stepwise_include_delimiter` | `false` | Include `;` in token spans |

## Output Format

The model is expected to output bounding boxes in **semicolon-delimited format**:

```
x1,y1,x2,y2,t;x1,y1,x2,y2,v;x1,y1,x2,y2,t;
```

where coordinates are in pixel values (e.g., 0-1000 range) and the last field is a type code: `t` = text, `v` = visual.

## Dependencies

- Python >= 3.9
- numpy
- scipy (for `linear_sum_assignment`)
- torch (for GDPO and trainer integration)

## License

Apache 2.0
