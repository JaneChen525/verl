"""Optional decision trace export for offline VLN credit case studies."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np


def sparse_01_advantages(rows: list[dict], epsilon: float = 1e-6) -> dict[tuple[object, object], float]:
    """Match ``grpo_trajectory`` on the same unique trajectory successes."""
    grouped: dict[object, dict[object, float]] = defaultdict(dict)
    for row in rows:
        if row.get("is_padding", False):
            continue
        uid = row["uid"]
        trajectory_uid = row["trajectory_uid"]
        value = float(row["trajectory_success"])
        previous = grouped[uid].get(trajectory_uid)
        if previous is not None and not np.isclose(previous, value):
            raise ValueError(f"inconsistent success for trajectory {trajectory_uid!r}")
        grouped[uid][trajectory_uid] = value

    result: dict[tuple[object, object], float] = {}
    for uid, trajectories in grouped.items():
        values = np.asarray(list(trajectories.values()), dtype=np.float64)
        if len(values) == 1:
            mean, std = 0.0, 1.0
        else:
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1))
        for trajectory_uid, value in trajectories.items():
            result[(uid, trajectory_uid)] = float((value - mean) / (std + epsilon))
    return result


def _builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _termination_reason(rows: list[dict]) -> str:
    last = max(rows, key=lambda row: int(row["turn_id"]))
    metrics = last.get("trajectory_metrics", {})
    if last.get("is_stop_action", False):
        return "stop_action"
    if not last.get("atomic_actions", []):
        return "empty_action"
    if int(metrics.get("env_steps", 0)) >= 200:
        return "max_env_steps"
    if int(metrics.get("num_decisions", 0)) >= 64:
        return "max_decisions"
    return "episode_over"


def export_decision_trace(rows: list[dict], meta_info: dict, path: str) -> None:
    """Write exact 0-1, P15, and P16 decision values as one JSONL file."""
    real_rows = [row for row in rows if not row.get("is_padding", False)]
    sparse_advantages = sparse_01_advantages(real_rows)

    trajectory_rows: dict[tuple[object, object], list[dict]] = defaultdict(list)
    rollout_index: dict[tuple[object, object], int] = {}
    next_index: dict[object, int] = defaultdict(int)
    for row in real_rows:
        key = (row["uid"], row["trajectory_uid"])
        trajectory_rows[key].append(row)
        if key not in rollout_index:
            rollout_index[key] = next_index[row["uid"]]
            next_index[row["uid"]] += 1

    termination = {
        key: _termination_reason(items) for key, items in trajectory_rows.items()
    }
    ordered = sorted(
        real_rows,
        key=lambda row: (
            str(row["uid"]),
            rollout_index[(row["uid"], row["trajectory_uid"])],
            int(row["turn_id"]),
        ),
    )

    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    global_step = int(meta_info.get("global_steps", -1))
    fields = (
        "action_text", "parsed_actions", "atomic_actions", "env_step_before",
        "env_step_after", "is_stop_action", "start_position", "end_position",
        "start_heading", "end_heading", "start_map_position", "end_map_position",
        "map_path", "atomic_rewards", "distance_path", "start_distance",
        "end_distance", "decision_reward", "discount_to_next", "decision_return", "credit_mode",
        "credit_region", "credit_weight", "buffer_distance", "buffer_support",
        "credit_criticality", "decision_loss_weight",
    )

    with temporary_path.open("w", encoding="utf-8") as handle:
        for row in ordered:
            key = (row["uid"], row["trajectory_uid"])
            record = {
                "schema_version": 1,
                "global_step": global_step,
                "uid": str(row["uid"]),
                "scene_id": str(row.get("scene_id", "")),
                "episode_id": str(row.get("episode_id", "")),
                "instruction": str(row.get("instruction", "")),
                "rollout_index": rollout_index[key],
                "trajectory_uid": str(row["trajectory_uid"]),
                "turn_id": int(row["turn_id"]),
                "trajectory_success": float(row["trajectory_success"]),
                "trajectory_reward": float(row["trajectory_reward"]),
                "trajectory_metrics": row.get("trajectory_metrics", {}),
                "termination_reason": termination[key],
                "advantage_01": sparse_advantages[key],
                "p15_return": float(row.get("decision_return", 0.0)),
                "p16_advantage": float(row["training_score"]),
            }
            for field in fields:
                record[field] = row.get(field)
            handle.write(json.dumps(_builtin(record), ensure_ascii=False) + "\n")
    temporary_path.replace(output_path)
    print(
        f"[VLN trace] {len(trajectory_rows)} trajectories, {len(real_rows)} decisions "
        f"→ {output_path}"
    )
