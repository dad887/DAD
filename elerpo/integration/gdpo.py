"""
GDPO: Group reward-Decoupled normalization Policy Optimization.

Reference: https://arxiv.org/abs/2601.05242

Instead of normalizing the sum of rewards (like GRPO), GDPO:
  1. Normalizes each reward function *separately* within generation groups.
  2. Sums the normalized advantages with per-function weights.
  3. Applies batch-wise normalization for stable magnitude.

This preserves more distinct advantage groups and enables more accurate
multi-reward optimization, which is important when combining visual-IoU
and text-IoU reward functions with different scales.

Integration: replace the ``_compute_advantages`` call in your GRPO trainer
with ``compute_gdpo_advantages`` when ``advantage_estimator == 'gdpo'``.
"""

import torch
from typing import List, Optional


def compute_gdpo_advantages(
    rewards_per_func: torch.Tensor,
    num_generations: int,
    reward_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Compute GDPO advantages.

    Args:
        rewards_per_func: (N, F) tensor of rewards per function, where N is
            the total number of samples and F is the number of reward functions.
            Samples are assumed to be grouped: samples ``[0..K-1]`` share the
            same prompt, ``[K..2K-1]`` the next, etc.
        num_generations: Number of generations per prompt (K).
        reward_weights: (F,) tensor of per-function weights.

    Returns:
        advantages: (N,) tensor of computed advantages.
    """
    N = rewards_per_func.size(0)
    K = num_generations
    num_funcs = rewards_per_func.size(1)

    # Step 1: per-reward group-wise normalization
    if K > 1:
        grouped = rewards_per_func.view(-1, K, num_funcs)  # [G, K, F]

        mean_per_func = torch.nanmean(grouped, dim=1, keepdim=True)  # [G, 1, F]
        diff = grouped - mean_per_func
        var_per_func = torch.nanmean(diff * diff, dim=1, keepdim=True)
        std_per_func = torch.sqrt(var_per_func + 1e-8)

        normalized = (grouped - mean_per_func) / std_per_func  # [G, K, F]
        advantages_per_func = normalized.view(N, num_funcs)
    else:
        mean_per_func = torch.nanmean(rewards_per_func, dim=0, keepdim=True)
        advantages_per_func = rewards_per_func - mean_per_func

    # Step 2: weighted sum
    weights = reward_weights.unsqueeze(0).expand(N, -1)
    A_sum = (advantages_per_func * weights).nansum(dim=1)  # [N]

    # Step 3: batch-wise normalization
    if A_sum.numel() > 1:
        A_mean = torch.nanmean(A_sum)
        A_diff = A_sum - A_mean
        A_var = torch.nanmean(A_diff * A_diff)
        A_std = torch.sqrt(A_var + 1e-8)
        advantages = (A_sum - A_mean) / A_std
    else:
        advantages = A_sum

    return torch.nan_to_num(advantages, nan=0.0)
