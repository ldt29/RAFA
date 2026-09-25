"""
Post-training module for GRPO-style RL fine-tuning.

Submodules:
    reward_fns       — Tier 1/2/3 reward functions (fnat proxy, CA-DockQ proxy, full DockQ)
    diversity_monitor — Sequence diversity tracking and collapse detection
    grpo_trainer     — Main GRPO training loop (generate → reward → advantage-weighted FM update)
"""