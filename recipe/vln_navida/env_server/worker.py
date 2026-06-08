"""Habitat env worker — one process, one Habitat Env, one active session at a time.

IPC: parent sends dict commands via cmd_queue, worker sends dict responses via resp_queue.
Commands:
  {"op": "reset",    "episode_id": str, "session_id": str}
  {"op": "step",     "actions": [int, ...]}
  {"op": "metrics"}
  {"op": "close"}
Worker sends {"ok": bool, ...payload...} or {"ok": False, "error": str}.
"""
import base64
import io
import multiprocessing as mp
import os
import time
import traceback

import numpy as np
from PIL import Image


def _encode_jpeg_b64(rgb: np.ndarray) -> str:
    img = Image.fromarray(rgb.astype("uint8")).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _make_obs(rgb: np.ndarray, instruction: str, step: int) -> dict:
    h, w = rgb.shape[:2]
    return {"rgb_jpeg_base64": _encode_jpeg_b64(rgb), "width": w, "height": h, "step": step,
            "instruction": instruction}


def _make_metrics(info: dict, collisions: int) -> dict:
    ne = float(info.get("distance_to_goal", 0.0))
    return {
        "distance_to_goal": ne,
        "success": float(info.get("success", 0.0)),
        "spl": float(info.get("spl", 0.0)),
        "oracle_success": float(info.get("oracle_success", 0.0)),
        "oracle_navigation_error": ne,
        "collisions": float(collisions),
    }


def _load_env(exp_config_path: str):
    import habitat
    from habitat import Env
    from habitat.config.default import get_config
    from habitat.config.default_structured_configs import (
        CollisionsMeasurementConfig, FogOfWarConfig, TopDownMapMeasurementConfig,
    )
    from habitat_extensions import measures, task  # noqa: F401  — registers oracle_success + VLN task

    config = get_config(exp_config_path)
    with habitat.config.read_write(config):
        config.habitat.task.measurements.update({
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3, map_resolution=1024,
                draw_source=True, draw_border=True, draw_shortest_path=True,
                draw_view_points=True, draw_goal_positions=True, draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(draw=True, visibility_dist=5.0, fov=90),
            ),
            "collisions": CollisionsMeasurementConfig(),
        })
    dataset = habitat.datasets.make_dataset(
        id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)
    env = Env(config=config, dataset=dataset)
    episodes_by_id = {str(ep.episode_id): ep for ep in dataset.episodes}
    return env, episodes_by_id


def worker_loop(worker_id: int, exp_config_path: str,
                cmd_queue: mp.Queue, resp_queue: mp.Queue, heartbeat_value,
                gpu_id: int = -1):
    try:
        if gpu_id >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env, episodes_by_id = _load_env(exp_config_path)
        resp_queue.put({"ok": True, "msg": f"worker {worker_id} ready ({len(episodes_by_id)} episodes)"})
    except Exception:
        resp_queue.put({"ok": False, "error": traceback.format_exc()})
        return

    step_count = 0
    collisions = 0
    current_episode = None

    while True:
        heartbeat_value.value = time.time()
        cmd = cmd_queue.get()
        op = cmd.get("op")
        try:
            if op == "close":
                env.close()
                resp_queue.put({"ok": True})
                return

            if op == "reset":
                ep_id = str(cmd["episode_id"])
                if ep_id not in episodes_by_id:
                    resp_queue.put({"ok": False, "error": f"episode_id {ep_id} not found"})
                    continue
                ep = episodes_by_id[ep_id]
                env.current_episode = ep
                obs = env.reset()
                current_episode = ep
                step_count = 0
                collisions = 0
                scene_id = os.path.basename(ep.scene_id).split(".")[0]
                info = env.get_metrics()
                resp_queue.put({
                    "ok": True,
                    "obs": _make_obs(obs["rgb"], obs["instruction"]["text"], step_count),
                    "metrics": _make_metrics(info, collisions),
                    "scene_id": scene_id,
                    "episode_id": ep_id,
                    "instruction": obs["instruction"]["text"],
                    "done": env.episode_over,
                })
                continue

            if op == "step":
                actions = cmd["actions"]
                last_obs = None
                done = False
                done_reason = None
                for action in actions:
                    obs = env.step({"action": int(action)})
                    last_obs = obs
                    step_count += 1
                    done = env.episode_over
                    if action == 0:
                        done_reason = "stop_action"
                    if done:
                        done_reason = done_reason or "episode_over"
                        break
                info = env.get_metrics()
                raw_col = info.get("collisions", {})
                collisions = int(raw_col.get("count", 0)) if isinstance(raw_col, dict) else 0
                resp_queue.put({
                    "ok": True,
                    "obs": _make_obs(last_obs["rgb"], last_obs["instruction"]["text"], step_count),
                    "metrics": _make_metrics(info, collisions),
                    "done": done,
                    "done_reason": done_reason,
                    "executed_actions": actions[:actions.index(0) + 1] if (0 in actions and done_reason == "stop_action") else actions,
                })
                continue

            if op == "metrics":
                info = env.get_metrics() if current_episode is not None else {}
                resp_queue.put({"ok": True, "metrics": _make_metrics(info, collisions),
                                "done": env.episode_over if current_episode else False})
                continue

            if op == "episodes":
                limit = int(cmd.get("limit", 200))
                resp_queue.put({"ok": True, "episode_ids": list(episodes_by_id.keys())[:limit]})
                continue

            resp_queue.put({"ok": False, "error": f"unknown op: {op}"})

        except Exception:
            resp_queue.put({"ok": False, "error": traceback.format_exc()})
