"""Trajectory reward (design doc §11). Stage 1 = sparse outcome (success).

metrics dict comes from VLNEnv.metrics() (keys match server StepResponse.metrics).
progress_coef=0 → pure SR; set >0 for oracle_success shaping when reward is too sparse.
"""


def compute_trajectory_reward(metrics: dict, progress_coef: float = 0.0) -> float:
    # P13: binary success reward.
    # reward = float(metrics.get("success", 0.0))
    # P14: success weighted by path efficiency.
    reward = float(metrics.get("spl", 0.0))
    if progress_coef:
        reward += progress_coef * float(metrics.get("oracle_success", 0.0))
    return reward
