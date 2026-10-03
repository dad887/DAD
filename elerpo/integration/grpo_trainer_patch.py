"""
Trainer modifications for EleRPO.

This file shows the exact changes needed in ms-swift's ``GRPOTrainer``
(``swift/trainers/rlhf_trainer/grpo_trainer.py``) to enable EleRPO
(element-level per-token advantage assignment).

There are four integration points:

  1. **Import** the stepwise module.
  2. **Initialize** EleRPO components in ``__init__``.
  3. **Route** to the hybrid scoring path in ``_generate_and_score_completions``.
  4. **Handle 2D advantages** in the loss computation.

The hybrid scoring method (``_generate_and_score_completions_hybrid``) is
provided in full below as it is the main new code.
"""

# ==========================================================================
# 1. IMPORT (add near the top of grpo_trainer.py)
# ==========================================================================

# --- begin patch: imports ---
try:
    from stepwise import (
        HungarianSlicing,
        SegmentIoUF1Reward,
        compute_batch_segment_advantages,
        build_per_token_advantages,
    )
    STEPWISE_AVAILABLE = True
except ImportError:
    STEPWISE_AVAILABLE = False
# --- end patch: imports ---


# ==========================================================================
# 2. INITIALIZATION (add at the end of GRPOTrainer.__init__)
# ==========================================================================

def _init_stepwise_training(self):
    """Initialize stepwise training components if enabled.

    Call this at the end of ``GRPOTrainer.__init__``.
    """
    args = self.args
    self.stepwise_training = getattr(args, "stepwise_training", False)

    if not self.stepwise_training:
        self.slicing_strategy = None
        self.stepwise_reward = None
        return

    if not STEPWISE_AVAILABLE:
        raise ImportError(
            "Stepwise training requires the stepwise package. "
            "Install it or ensure it is on PYTHONPATH."
        )

    iou_threshold = getattr(args, "stepwise_iou_threshold", 0.5)
    include_delimiter = getattr(args, "stepwise_include_delimiter", False)

    self.slicing_strategy = HungarianSlicing(
        iou_threshold=iou_threshold,
        include_delimiter_in_span=include_delimiter,
    )

    self.stepwise_reward = SegmentIoUF1Reward(
        iou_threshold=iou_threshold,
        use_iou_as_reward=getattr(args, "stepwise_use_iou_reward", True),
        check_labels=getattr(args, "stepwise_check_labels", True),
    )

    self.stepwise_norm_mode = getattr(args, "stepwise_norm_mode", "batch")
    self.stepwise_im_end_recall = getattr(args, "stepwise_im_end_recall", False)
    self.stepwise_delta_f1_weights = getattr(args, "stepwise_delta_f1_weights", None)
    self.stepwise_n_clusters = getattr(args, "stepwise_n_clusters", 2)
    self.stepwise_min_silhouette = getattr(args, "stepwise_min_silhouette", 0.5)
    self.stepwise_hybrid_mode = getattr(args, "stepwise_hybrid_mode", "average")


# ==========================================================================
# 3. ROUTING (modify _generate_and_score_completions)
# ==========================================================================

def _generate_and_score_completions_patched(self, inputs):
    """
    Replace the body of ``_generate_and_score_completions`` with this.

    The only change is the ``if self.stepwise_training`` branch that routes
    to the hybrid method when sequence-level reward functions are available.
    """
    inputs = self._generate_completions(inputs)

    # --- begin patch: stepwise routing ---
    if self.stepwise_training and self.reward_funcs:
        return self._generate_and_score_completions_hybrid(inputs)
    # --- end patch ---

    # ... original scoring logic continues unchanged ...


# ==========================================================================
# 4. HYBRID SCORING METHOD (add to GRPOTrainer)
# ==========================================================================

def _generate_and_score_completions_hybrid(self, inputs):
    """
    Hybrid mode: combines sequence-level GRPO/GDPO advantages with
    per-token box-level advantages.

    sequence_advantage is broadcast to every token in a sample.
    box_level_advantage varies per token (from stepwise slicing).
    final_advantage[t] = 0.5 * seq_adv + 0.5 * box_adv[t]  (average mode)
    """
    import numpy as np
    import torch

    mode = "train" if self.model.training else "eval"

    # --- Part 1: sequence-level rewards and advantages (standard GRPO) ---
    total_rewards_per_func = self._score_completions(inputs)
    batch_encoded_inputs = self._prepare_batch_inputs(inputs)
    total_advantages = self._compute_advantages(
        inputs, total_rewards_per_func, batch_encoded_inputs
    )

    # --- Part 2: per-token box-level advantages (stepwise) ---
    all_slicing_results = []
    all_segment_rewards = []

    for inp in inputs:
        completion = inp["messages"][-1]["content"]
        if isinstance(completion, dict):
            completion = self.processing_class.decode(
                completion.get("token_ids", [])
            )
        elif isinstance(completion, list):
            completion = self.processing_class.decode(completion)

        solution = inp.get("solution", {})
        gt_boxes = np.array(solution.get("boxes", []), dtype=np.float32)
        gt_labels = solution.get("types", None)

        slicing_result = self.slicing_strategy.slice(
            completion, gt_boxes, gt_labels, self.processing_class
        )
        all_slicing_results.append(slicing_result)
        all_segment_rewards.append(
            self.stepwise_reward.compute_rewards(slicing_result)
        )

    # Gather across ranks (distributed training)
    from accelerate.utils import gather_object
    all_slicing_results_gathered = gather_object(all_slicing_results)
    all_segment_rewards_gathered = gather_object(all_segment_rewards)

    num_gen = (self.num_generations if mode == "train"
               else self.num_generations_eval)

    all_segment_advantages, all_recall_advantages, cluster_stats = \
        compute_batch_segment_advantages(
            all_slicing_results_gathered,
            all_segment_rewards_gathered,
            num_generations=num_gen,
            norm_mode=self.stepwise_norm_mode,
            im_end_recall=self.stepwise_im_end_recall,
            reward_weights=(
                self.reward_weights.tolist()
                if hasattr(self, "reward_weights") and self.reward_funcs
                else None
            ) or self.stepwise_delta_f1_weights,
            n_clusters=self.stepwise_n_clusters,
            min_silhouette=self.stepwise_min_silhouette,
        )

    # Log clustering diagnostics
    if cluster_stats is not None:
        self._metrics[mode]["cluster_k_mean"].append(
            cluster_stats["cluster_k_mean"])
        self._metrics[mode]["cluster_silhouette_mean"].append(
            cluster_stats["cluster_silhouette_mean"])
        self._metrics[mode]["cluster_used_frac"].append(
            cluster_stats["cluster_used_frac"])

    # Distribute back to local rank
    local_slicing_results = self._get_local_data(
        all_slicing_results_gathered
    )
    local_segment_advantages = self._get_local_data(
        all_segment_advantages
    )
    local_recall_advantages = (
        self._get_local_data(all_recall_advantages)
        if all_recall_advantages else None
    )

    for i, inp in enumerate(inputs):
        inp["slicing_result"] = local_slicing_results[i]
        inp["segment_advantages"] = local_segment_advantages[i]
        if local_recall_advantages is not None:
            inp["recall_advantage"] = local_recall_advantages[i]

    # --- Part 3: combine both advantages ---
    local_advantages = self._get_local_data(total_advantages)
    for i, inp in enumerate(inputs):
        inp["advantages"] = local_advantages[i]

    gas_chunks = self.split_by_mini_batches(inputs)

    for batch, batch_encoded in zip(gas_chunks, batch_encoded_inputs):
        batch_size = len(batch)
        seq_len = batch_encoded["completion_mask"].shape[-1]

        # Sequence-level: broadcast scalar to all tokens
        seq_advantages = torch.stack(
            [data["advantages"] for data in batch]
        )  # [B]
        seq_adv_broadcast = seq_advantages.unsqueeze(1).expand(
            batch_size, seq_len
        )

        # Element-level: per-token from stepwise
        box_adv = np.zeros((batch_size, seq_len), dtype=np.float32)
        for b, data in enumerate(batch):
            box_adv[b] = build_per_token_advantages(
                data["slicing_result"],
                data["segment_advantages"],
                seq_len,
            )
            # Assign recall advantage to im_end token
            use_im_end = (self.stepwise_im_end_recall
                          or self.stepwise_norm_mode.startswith("delta_f1"))
            if use_im_end and "recall_advantage" in data:
                completion_mask = batch_encoded["completion_mask"][b]
                nonzero = completion_mask.nonzero(as_tuple=True)[0]
                if len(nonzero) > 0:
                    box_adv[b][nonzero[-1].item()] = data["recall_advantage"]

        box_adv_tensor = torch.tensor(
            box_adv, dtype=torch.float32, device=self.accelerator.device
        )

        # Combine
        if self.stepwise_hybrid_mode == "element_only":
            batch_encoded["advantages"] = box_adv_tensor
        elif self.stepwise_hybrid_mode == "average":
            batch_encoded["advantages"] = (
                0.5 * seq_adv_broadcast + 0.5 * box_adv_tensor
            )

        # Signal to the loss function that advantages are per-token
        batch_encoded["stepwise_mode"] = True

    return batch_encoded_inputs


# ==========================================================================
# 5. LOSS FUNCTION CHANGE (modify _compute_loss / the loss computation)
# ==========================================================================
#
# In the GRPO loss computation, advantages are normally a 1D tensor of
# shape [batch_size] that gets broadcast to [batch_size, seq_len].
#
# When ``stepwise_mode=True``, advantages are already [batch_size, seq_len].
# Add this check before expanding advantages:
#
#   stepwise_mode = inputs.get('stepwise_mode', False)
#   if stepwise_mode:
#       advantages_expanded = advantages  # already [B, seq_len]
#   else:
#       advantages_expanded = advantages.unsqueeze(1)  # [B] -> [B, 1]
#
# The rest of the loss computation (clipped ratio * advantages * mask)
# remains unchanged.
