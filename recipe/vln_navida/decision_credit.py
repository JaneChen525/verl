"""Decision-level credit assignment from successful rollouts in one VLN group.

The helper is intentionally NumPy-only.  It mutates each decision row's
``training_score`` before the rollout manager builds the padded DataProto.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def _vector(row: dict, key: str, size: int) -> np.ndarray | None:
    value = np.asarray(row.get(key, []), dtype=np.float64)
    if value.shape != (size,) or not np.all(np.isfinite(value)):
        return None
    return value


def _heading(row: dict, key: str) -> np.ndarray | None:
    value = _vector(row, key, 2)
    if value is None:
        return None
    norm = float(np.linalg.norm(value))
    if norm <= 1e-8:
        return None
    return value / norm


def _p15_score(row: dict) -> float:
    return float(row.get("decision_return", row.get("training_score", 0.0)))


def _fallback(rows: list[dict]) -> None:
    for row in rows:
        row["training_score"] = _p15_score(row)
        row["credit_mode"] = "p15_fallback"
        row["credit_weight"] = 0.0


def _nearest_reference(
    position: np.ndarray,
    heading: np.ndarray,
    reference_positions: np.ndarray,
    reference_headings: np.ndarray,
    heading_neighbors: int = 8,
) -> tuple[float, float]:
    distances = np.linalg.norm(reference_positions - position[None, :], axis=1)
    count = min(heading_neighbors, len(distances))
    nearest = np.argpartition(distances, count - 1)[:count]
    max_cosine = float(np.max(reference_headings[nearest] @ heading))
    return float(np.min(distances)), float(np.clip(max_cosine, -1.0, 1.0))


def _persistent_exit(distances: np.ndarray, radius: float, horizon: int) -> int | None:
    """First decision followed by ``horizon`` consecutive out-of-buffer states."""
    for index in range(len(distances)):
        if index + horizon <= len(distances) and np.all(
            distances[index : index + horizon] > radius
        ):
            return index
    return None


def _stagnation_onset(rows: list[dict], min_progress: float) -> int:
    """Earliest decision after which the trajectory never improves by min_progress."""
    start_distances = np.asarray(
        [float(row.get("start_distance", np.nan)) for row in rows], dtype=np.float64
    )
    end_distances = np.asarray(
        [float(row.get("end_distance", np.nan)) for row in rows], dtype=np.float64
    )
    if not np.all(np.isfinite(start_distances)) or not np.all(np.isfinite(end_distances)):
        return len(rows) - 1
    future_best = np.minimum.accumulate(end_distances[::-1])[::-1]
    candidates = np.flatnonzero(start_distances - future_best <= min_progress)
    return int(candidates[0]) if len(candidates) else len(rows) - 1


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    logits = values / temperature
    logits -= np.max(logits)
    weights = np.exp(logits)
    return weights / np.sum(weights)


def assign_success_buffer_credit(
    rows: list[dict],
    *,
    radius: float = 1.5,
    direction_beta: float = 0.5,
    candidate_window: int = 3,
    exit_horizon: int = 2,
    temperature: float = 0.2,
    key_penalty: float = 1.0,
    reward_scale: float = 1.0,
    stagnation_progress: float = 0.25,
) -> dict[str, float]:
    """Assign P16 decision scores in place and return logging metrics.

    Rows are grouped by ``uid`` (one prompt) and then ``trajectory_uid``.
    Successful trajectories form the spatial/heading reference buffer.  Failed
    trajectories receive one unit of negative credit concentrated immediately
    before their persistent buffer exit.  A failed STOP inside the buffer is
    treated as the key decision.  Groups without a successful trajectory fall
    back exactly to the P15 decision return.
    """
    if radius <= 0 or candidate_window <= 0 or exit_horizon <= 0:
        raise ValueError("radius, candidate_window, and exit_horizon must be positive")
    if temperature <= 0 or reward_scale <= 0 or key_penalty < 0 or stagnation_progress < 0:
        raise ValueError(
            "temperature/reward_scale must be positive; key_penalty/stagnation_progress non-negative"
        )

    grouped: dict[object, dict[object, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if not row.get("is_padding", False):
            grouped[row["uid"]][row["trajectory_uid"]].append(row)
    for trajectories in grouped.values():
        for trajectory_rows in trajectories.values():
            trajectory_rows.sort(key=lambda row: int(row["turn_id"]))

    fallback_groups = 0
    failed_trajectories = 0
    early_stops = 0
    persistent_exits = 0
    stagnation_trajectories = 0
    assigned_scores: list[float] = []
    prefix_scores: list[float] = []
    key_scores: list[float] = []
    post_scores: list[float] = []

    for trajectories in grouped.values():
        success_trajectory_ids = {
            trajectory_uid
            for trajectory_uid, trajectory_rows in trajectories.items()
            if trajectory_rows
            and float(trajectory_rows[0].get("trajectory_success", 0.0)) > 0.5
        }
        success_trajectories = [
            trajectories[trajectory_uid] for trajectory_uid in success_trajectory_ids
        ]
        group_rows = [row for trajectory_rows in trajectories.values() for row in trajectory_rows]
        if not success_trajectories:
            _fallback(group_rows)
            fallback_groups += 1
            continue

        reference_positions: list[np.ndarray] = []
        reference_headings: list[np.ndarray] = []
        valid_reference = True
        for trajectory_rows in success_trajectories:
            for row in trajectory_rows:
                for position_key, heading_key in (
                    ("start_position", "start_heading"),
                    ("end_position", "end_heading"),
                ):
                    position = _vector(row, position_key, 3)
                    heading = _heading(row, heading_key)
                    if position is None or heading is None:
                        valid_reference = False
                        break
                    reference_positions.append(position)
                    reference_headings.append(heading)
                if not valid_reference:
                    break
            if not valid_reference:
                break
        if not valid_reference or not reference_positions:
            _fallback(group_rows)
            fallback_groups += 1
            continue

        ref_positions = np.stack(reference_positions)
        ref_headings = np.stack(reference_headings)

        group_invalid = False
        failed_payloads = []
        for trajectory_uid, trajectory_rows in trajectories.items():
            if trajectory_uid in success_trajectory_ids:
                continue
            starts, ends, start_headings, end_headings = [], [], [], []
            for row in trajectory_rows:
                starts.append(_vector(row, "start_position", 3))
                ends.append(_vector(row, "end_position", 3))
                start_headings.append(_heading(row, "start_heading"))
                end_headings.append(_heading(row, "end_heading"))
            if any(value is None for value in starts + ends + start_headings + end_headings):
                group_invalid = True
                break
            failed_payloads.append((trajectory_rows, starts, ends, start_headings, end_headings))
        if group_invalid:
            # Do not silently train on partially malformed pose data.
            _fallback(group_rows)
            fallback_groups += 1
            continue

        # Successful decisions have zero buffer distance, so Phi=1 and no key penalty.
        for trajectory_rows in success_trajectories:
            for row in trajectory_rows:
                score = float(np.clip(float(row.get("decision_reward", 0.0)) / reward_scale, -1.0, 1.0))
                row.update(
                    training_score=score,
                    credit_mode="success_buffer",
                    credit_weight=0.0,
                    buffer_distance=0.0,
                    buffer_support=1.0,
                    credit_criticality=0.0,
                    credit_region="success",
                )
                assigned_scores.append(score)

        for trajectory_rows, starts, ends, start_headings, end_headings in failed_payloads:
            failed_trajectories += 1
            start_distances, end_distances, criticality = [], [], []
            for start, end, start_heading, end_heading in zip(
                starts, ends, start_headings, end_headings
            ):
                start_distance, start_cosine = _nearest_reference(
                    start, start_heading, ref_positions, ref_headings
                )
                end_distance, end_cosine = _nearest_reference(
                    end, end_heading, ref_positions, ref_headings
                )
                start_distances.append(start_distance)
                end_distances.append(end_distance)
                criticality.append(
                    max(0.0, end_distance - start_distance) / radius
                    + direction_beta
                    * max(0.0, (1.0 - end_cosine) / 2.0 - (1.0 - start_cosine) / 2.0)
                )

            start_distances_np = np.asarray(start_distances)
            end_distances_np = np.asarray(end_distances)
            criticality_np = np.asarray(criticality)
            supports = np.exp(-(start_distances_np**2) / (2.0 * radius**2))
            local_scores = supports * np.clip(
                np.asarray([float(row.get("decision_reward", 0.0)) for row in trajectory_rows])
                / reward_scale,
                -1.0,
                1.0,
            )

            stop_candidates = [
                index
                for index, row in enumerate(trajectory_rows)
                if row.get("is_stop_action", False)
                and max(start_distances[index], end_distances[index]) <= radius
            ]
            weights = np.zeros(len(trajectory_rows), dtype=np.float64)
            if stop_candidates:
                key_index = stop_candidates[-1]
                exit_index = key_index
                weights[key_index] = 1.0
                early_stops += 1
            else:
                exit_index = _persistent_exit(end_distances_np, radius, exit_horizon)
                if exit_index is not None:
                    persistent_exits += 1
                else:
                    exit_index = _stagnation_onset(trajectory_rows, stagnation_progress)
                    stagnation_trajectories += 1
                first_candidate = max(0, exit_index - candidate_window + 1)
                candidates = np.arange(first_candidate, exit_index + 1)
                if np.max(criticality_np[candidates]) <= 1e-8:
                    weights[exit_index] = 1.0
                else:
                    weights[candidates] = _softmax(criticality_np[candidates], temperature)
                key_index = int(np.argmax(weights))

            # Decisions after the identified failure cannot recover this rollout;
            # mask their tiny exploration rewards so a long max-step tail adds no votes.
            local_scores[exit_index + 1 :] = 0.0
            scores = local_scores - key_penalty * weights
            first_candidate = max(0, exit_index - candidate_window + 1)
            for index, (row, score) in enumerate(zip(trajectory_rows, scores)):
                if index > exit_index:
                    region = "post"
                    post_scores.append(float(score))
                elif index >= first_candidate:
                    region = "key" if index == key_index else "candidate"
                    if index == key_index:
                        key_scores.append(float(score))
                else:
                    region = "prefix"
                    prefix_scores.append(float(score))
                row.update(
                    training_score=float(score),
                    credit_mode="success_buffer",
                    credit_weight=float(weights[index]),
                    buffer_distance=float(start_distances[index]),
                    buffer_support=float(supports[index]),
                    credit_criticality=float(criticality[index]),
                    credit_region=region,
                )
                assigned_scores.append(float(score))

    group_count = len(grouped)
    metrics = {
        "vln/credit/groups": float(group_count),
        "vln/credit/fallback_groups": float(fallback_groups),
        "vln/credit/fallback_group_ratio": float(fallback_groups / group_count) if group_count else 0.0,
        "vln/credit/failed_trajectories": float(failed_trajectories),
        "vln/credit/early_stop_trajectories": float(early_stops),
        "vln/credit/persistent_exit_trajectories": float(persistent_exits),
        "vln/credit/stagnation_trajectories": float(stagnation_trajectories),
    }
    for name, values in (
        ("score_mean", assigned_scores),
        ("prefix_score_mean", prefix_scores),
        ("key_score_mean", key_scores),
        ("post_abs_score_mean", [abs(value) for value in post_scores]),
    ):
        metrics[f"vln/credit/{name}"] = float(np.mean(values)) if values else 0.0
    return metrics
