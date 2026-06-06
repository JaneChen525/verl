"""Trajectory-level GRPO advantage (design doc §10).

P3 MVP: uses standard GRPO (groups by uid = episode start). All decisions of a
trajectory share the same rm_scores[-1] = trajectory_reward. Standard GRPO computes
(reward - group_mean) / std, which is length-weighted (longer trajectories have
more "votes" in the mean). decision_loss_weight (1/num_decisions, set by manager)
partially addresses this at the loss level.

TODO (post-P3): aggregate by trajectory_uid first (so each trajectory contributes
exactly one reward to the group mean/std), then broadcast to decisions. This gives
true trajectory-level GRPO per §10.2.
"""
