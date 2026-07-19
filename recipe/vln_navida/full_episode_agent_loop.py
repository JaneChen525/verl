"""VLN full-episode rollout (design doc §8.3).

run_episode() drives one Habitat trajectory to done/cap and records every
NaVIDA decision as a DecisionRecord. The only backend-specific piece is decide():

  P1 standalone (rollout_smoke):
    b64_buffer -> build_navida_messages_b64 -> vLLM OpenAI API -> action_text
    decide returns DecisionGen(action_text=...) only (no token ids).

  P2+ verl (VLNFullEpisodeAgentLoop):
    b64_buffer -> apply_chat_template -> server_manager.generate
    decide returns DecisionGen with prompt_ids / response_ids / response_mask / images
    for training. The same episode loop is reused unchanged.
"""
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from recipe.vln_navida.action import parse_navida_action, to_atomic_chunk
from recipe.vln_navida.reward import compute_trajectory_reward

MAX_ACTION_HISTORY = 200  # frame buffer cap (matches eval --max-action-history 200)


@dataclass
class DecisionGen:
    """What a backend returns for one LLM call. Training fields are None in P1."""
    action_text: str
    prompt_ids: Optional[list[int]] = None
    response_ids: Optional[list[int]] = None
    response_logprobs: Optional[list[float]] = None
    response_mask: Optional[list[int]] = None
    images: Optional[list[Any]] = None           # PIL images (rollout inference only)
    mm_processor_kwargs: Optional[dict] = None
    image_indices: Optional[list[int]] = None    # indices into episode b64_buffer
    raw_prompt: Optional[str] = None             # pre-tokenization prompt string
    position_ids: Optional[list[list[int]]] = None  # full prompt+response, axes x sequence


@dataclass
class DecisionRecord:
    turn_id: int
    action_text: str
    parsed_actions: list
    atomic_chunk: list[int]
    env_step_before: int
    env_step_after: int
    is_stop_action: bool
    gen: DecisionGen
    decision_reward: float = 0.0
    discount_to_next: float = 1.0
    decision_return: float = 0.0
    start_position: list[float] = field(default_factory=list)
    start_heading: list[float] = field(default_factory=list)
    end_position: list[float] = field(default_factory=list)
    end_heading: list[float] = field(default_factory=list)
    start_map_position: list[float] = field(default_factory=list)
    end_map_position: list[float] = field(default_factory=list)
    map_path: list[list[float]] = field(default_factory=list)
    atomic_rewards: list[float] = field(default_factory=list)
    distance_path: list[float] = field(default_factory=list)
    start_distance: float = 0.0
    end_distance: float = 0.0


@dataclass
class TrajectoryRecord:
    group_uid: str
    trajectory_uid: str
    scene_id: str
    episode_id: str
    instruction: str
    reward: float
    metrics: dict
    decisions: list[DecisionRecord] = field(default_factory=list)
    image_buffer: list[str] = field(default_factory=list)


async def run_episode(
    env,                                              # VLNEnv instance (already reset externally, OR reset here)
    extra_info: dict,
    decide: Callable[..., Awaitable[DecisionGen]],
    *,
    group_uid: str,
    trajectory_uid: str,
    max_decisions: int = 64,
    max_env_steps: int = 200,                        # P7 target (report/020)
    progress_coef: float = 0.0,
    reward_mode: str = "sparse_sr",
    dense_gamma: float = 0.95,
) -> TrajectoryRecord:
    """Drive one full episode. env must NOT be reset yet; this function resets it.

    decide(instruction, b64_buffer) -> DecisionGen
      b64_buffer: list of JPEG base64 strings, [-1] = current frame.
      b64_buffer is accumulated across turns (capped at MAX_ACTION_HISTORY).
    """
    b64 = await env.reset(extra_info)
    all_frames: list[str] = [b64]
    history_window: list[str] = [b64]
    decisions: list[DecisionRecord] = []
    env_steps = 0
    done = False
    dense_enabled = reward_mode == "p15_dense"
    if reward_mode not in {"sparse_sr", "p15_dense"}:
        raise ValueError(f"Unsupported VLN reward mode: {reward_mode}")
    if dense_enabled and not 0.0 <= dense_gamma <= 1.0:
        raise ValueError(f"dense_gamma must be in [0, 1], got {dense_gamma}")
    previous_distance = float(env.metrics().get("distance_to_goal", 0.0))

    while not done and len(decisions) < max_decisions and env_steps < max_env_steps:
        start_metrics = env.metrics()
        start_position = [float(x) for x in start_metrics.get("position", [])]
        start_heading = [float(x) for x in start_metrics.get("heading", [])]
        start_map_position = [float(x) for x in start_metrics.get("map_position", [])]
        start_distance = float(start_metrics.get("distance_to_goal", 0.0))
        gen = await decide(env.instruction, history_window, all_frames)
        parsed = parse_navida_action(gen.action_text)
        chunk = to_atomic_chunk(parsed)
        step_before = env_steps
        decision_reward = 0.0
        discount_to_next = 1.0
        executed_chunk: list[int] = []
        map_path = [start_map_position] if start_map_position else []
        atomic_rewards: list[float] = []
        distance_path = [start_distance]

        if not chunk:
            decisions.append(DecisionRecord(
                turn_id=len(decisions), action_text=gen.action_text,
                parsed_actions=parsed, atomic_chunk=[], env_step_before=env_steps,
                env_step_after=env_steps, is_stop_action=False, gen=gen,
                start_position=start_position, start_heading=start_heading,
                end_position=start_position, end_heading=start_heading,
                start_map_position=start_map_position,
                end_map_position=start_map_position,
                map_path=map_path,
                atomic_rewards=atomic_rewards,
                distance_path=distance_path,
                start_distance=start_distance, end_distance=start_distance,
            ))
            break

        for atomic in chunk:
            resp = await env.step([atomic])
            executed_chunk.extend(int(action) for action in resp.get("executed_actions", [atomic]))
            frame = env.current_jpeg_b64()
            all_frames.append(frame)
            history_window.append(frame)
            if len(history_window) > MAX_ACTION_HISTORY:
                history_window = history_window[1:]
            env_steps += 1
            done = resp["done"]
            step_metrics = resp["metrics"]
            current_distance = float(step_metrics["distance_to_goal"])
            map_position = [float(x) for x in step_metrics.get("map_position", [])]
            if map_position:
                map_path.append(map_position)
            distance_path.append(current_distance)
            if dense_enabled:
                atomic_reward = previous_distance - current_distance - 0.01
                if done:
                    atomic_reward += 2.5 * float(step_metrics.get("success", 0.0))
                atomic_rewards.append(atomic_reward)
                decision_reward += discount_to_next * atomic_reward
                discount_to_next *= dense_gamma
                previous_distance = current_distance
            if done:
                break

        end_metrics = env.metrics()
        decisions.append(DecisionRecord(
            turn_id=len(decisions), action_text=gen.action_text,
            parsed_actions=parsed, atomic_chunk=executed_chunk,
            env_step_before=step_before, env_step_after=env_steps,
            is_stop_action=(0 in executed_chunk), gen=gen,
            decision_reward=decision_reward,
            discount_to_next=discount_to_next,
            start_position=start_position,
            start_heading=start_heading,
            end_position=[float(x) for x in end_metrics.get("position", [])],
            end_heading=[float(x) for x in end_metrics.get("heading", [])],
            start_map_position=start_map_position,
            end_map_position=[float(x) for x in end_metrics.get("map_position", [])],
            map_path=map_path,
            atomic_rewards=atomic_rewards,
            distance_path=distance_path,
            start_distance=start_distance,
            end_distance=float(end_metrics.get("distance_to_goal", 0.0)),
        ))

    if dense_enabled:
        running_return = 0.0
        for decision in reversed(decisions):
            running_return = (
                decision.decision_reward + decision.discount_to_next * running_return
            )
            decision.decision_return = running_return

    metrics = env.metrics()
    metrics["num_decisions"] = len(decisions)
    metrics["env_steps"] = env_steps
    return TrajectoryRecord(
        group_uid=group_uid, trajectory_uid=trajectory_uid,
        scene_id=str(extra_info.get("scene_id", "")),
        episode_id=str(extra_info["episode_id"]),
        instruction=env.instruction,
        reward=compute_trajectory_reward(metrics, progress_coef=progress_coef),
        metrics=metrics,
        decisions=decisions,
        image_buffer=all_frames,
    )
