"""VLNOnlineRolloutManager — replaces AgentLoopManager for full-episode flatten (P2, design doc §8.4).

Standard flow:  1 row → 1 AgentLoopOutput → 1 training sample.
VLN flow:       1 row → 1 full episode → N decisions → N training samples (flatten).

This manager inherits AgentLoopManager and overrides generate_sequences:
1. Splits rollouts into windows (rollout_window controls concurrency).
2. For each window, calls super() → 1:1 DataProto (each row = one episode-trial).
3. Extracts trajectory data (all decisions) from extra_fields["trajectory"].
4. Flattens: each decision → one padded row (prompt/response/mask/rm_scores/position_ids).
5. Returns the flattened DataProto (rows = total decisions across all trajectories).

The trainer receives a batch where each row is an INDEPENDENT NaVIDA decision, and
uid groups all decisions of the same episode start for trajectory-level GRPO (§10).
"""
import asyncio
import base64
import io
import os

import numpy as np
import ray
import torch
from PIL import Image

from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.protocol import DataProto
from verl.utils.ray_utils import auto_await
from verl.utils.tokenizer import build_multimodal_processor_inputs


def _b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

# ── Monkey-patch DataProto.union to handle VLN flatten ────────────────────────
# When VLNOnlineRolloutManager returns a flattened batch (M decision rows) and
# the trainer tries batch.union(gen_output) with mismatched sizes (N vs M),
# skip the union and use the flattened output directly. Subsequent unions
# (reward, logprob) have matching sizes and work normally.
_original_union = DataProto.union

def _vln_union(self, other):
    if getattr(other, "meta_info", None) and other.meta_info.get("vln_flattened"):
        # Carry over meta_info keys the trainer expects (set on the original batch)
        for key in ("temperature", "eos_token_id", "pad_token_id", "recompute_log_prob",
                     "do_sample", "validate", "global_steps"):
            if hasattr(self, "meta_info") and key in self.meta_info and key not in other.meta_info:
                other.meta_info[key] = self.meta_info[key]
        return other
    return _original_union(self, other)

DataProto.union = _vln_union


class VLNOnlineRolloutManager(AgentLoopManager):
    """Full-episode rollout + flatten decisions, with rollout_window concurrency control."""

    def _get_tokenizer_and_processor(self):
        """Lazy-load tokenizer + multimodal processor (for computing pixel_values in flatten)."""
        if not hasattr(self, "_tokenizer_cached"):
            from transformers import AutoProcessor, AutoTokenizer
            model_path = self.model_config.path
            self._tokenizer_cached = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self._processor_cached = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        return self._tokenizer_cached, self._processor_cached

    def _get_rollout_window(self, total: int) -> int:
        """Read VLN_ROLLOUT_WINDOW from env var; default = total (no windowing)."""
        w = os.environ.get("VLN_ROLLOUT_WINDOW")
        if w is not None:
            try:
                w = int(w)
                if w > 0:
                    return w
            except ValueError:
                pass
        return total

    def _extract_decisions(self, one_to_one: DataProto):
        """Extract decision rows + reward/success from 1:1 rollout output.

        Returns (all_rows: list[dict], traj_rewards: list[float], traj_successes: list[float]).
        """
        trajectories = one_to_one.non_tensor_batch.get("trajectory")
        if trajectories is None:
            return [], [], []

        all_rows = []
        traj_rewards = []
        traj_successes = []
        for traj_data in trajectories:
            if traj_data is None:
                continue
            group_uid = traj_data["group_uid"]
            trajectory_uid = traj_data["trajectory_uid"]
            reward = traj_data["reward"]
            num_decisions = traj_data["num_decisions"]
            decisions = traj_data.get("decisions", [])
            image_buffer = traj_data.get("image_buffer", [])
            image_buffer_ref = ray.put(image_buffer) if image_buffer else None
            traj_rewards.append(reward)
            metrics = traj_data.get("metrics", {})
            traj_successes.append(float(metrics.get("success", 0.0)))

            for dec in decisions:
                all_rows.append({
                    "prompt_ids": dec["prompt_ids"],
                    "response_ids": dec["response_ids"],
                    "response_logprobs": dec.get("response_logprobs"),
                    "response_mask": dec["response_mask"],
                    "image_buffer_ref": image_buffer_ref,
                    "image_indices": dec.get("image_indices"),
                    "raw_prompt": dec.get("raw_prompt"),
                    "mm_processor_kwargs": dec.get("mm_processor_kwargs"),
                    "position_ids": dec.get("position_ids"),
                    "turn_id": dec["turn_id"],
                    "action_text": dec.get("action_text", ""),
                    "is_stop_action": dec.get("is_stop_action", False),
                    "uid": group_uid,
                    "trajectory_uid": trajectory_uid,
                    "trajectory_reward": reward,
                    "trajectory_success": float(metrics.get("success", 0.0)),
                    "decision_reward": float(dec.get("decision_reward", 0.0)),
                    "decision_return": float(dec.get("decision_return", 0.0)),
                    "start_position": dec.get("start_position", []),
                    "start_heading": dec.get("start_heading", []),
                    "end_position": dec.get("end_position", []),
                    "end_heading": dec.get("end_heading", []),
                    "start_distance": float(dec.get("start_distance", 0.0)),
                    "end_distance": float(dec.get("end_distance", 0.0)),
                    "training_score": dec.get("training_score", reward),
                    "decision_loss_weight": 1.0 / max(num_decisions, 1),
                })

        return all_rows, traj_rewards, traj_successes

    def _build_output(self, all_rows, traj_rewards, traj_successes, meta_info) -> DataProto:
        """Build padded DataProto from collected decision rows."""
        credit_mode = os.environ.get("VLN_CREDIT_MODE", "off").lower()
        credit_metrics = {}
        if credit_mode == "success_buffer":
            if os.environ.get("VLN_REWARD_MODE", "sparse_sr") != "p15_dense":
                raise ValueError(
                    "VLN_CREDIT_MODE=success_buffer requires VLN_REWARD_MODE=p15_dense"
                )
            from recipe.vln_navida.decision_credit import assign_success_buffer_credit

            credit_metrics = assign_success_buffer_credit(
                all_rows,
                radius=float(os.environ.get("VLN_CREDIT_RADIUS", "1.5")),
                direction_beta=float(os.environ.get("VLN_CREDIT_DIRECTION_BETA", "0.5")),
                candidate_window=int(os.environ.get("VLN_CREDIT_WINDOW", "3")),
                exit_horizon=int(os.environ.get("VLN_CREDIT_HORIZON", "2")),
                temperature=float(os.environ.get("VLN_CREDIT_TEMPERATURE", "0.2")),
                key_penalty=float(os.environ.get("VLN_CREDIT_KEY_PENALTY", "1.0")),
                reward_scale=float(os.environ.get("VLN_CREDIT_REWARD_SCALE", "1.0")),
                stagnation_progress=float(
                    os.environ.get("VLN_CREDIT_STAGNATION_PROGRESS", "0.25")
                ),
            )
        elif credit_mode not in {"", "off", "none"}:
            raise ValueError(f"Unsupported VLN credit mode: {credit_mode}")

        prompt_length = self.rollout_config.prompt_length
        response_length = self.rollout_config.response_length

        # Pad to multiple of (fsdp_size * micro_batch_size)
        try:
            fsdp_size = self.config.actor_rollout_ref.actor.fsdp_config.fsdp_size
        except Exception:
            fsdp_size = 8
        try:
            micro_bs = max(
                self.config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                self.config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
            )
        except Exception:
            micro_bs = 2
        try:
            ppo_mbs = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            rollout_n = self.config.actor_rollout_ref.rollout.n
        except Exception:
            ppo_mbs, rollout_n = 32, 4
        pad_multiple = max(fsdp_size * micro_bs, ppo_mbs * rollout_n)
        real_decision_count = len(all_rows)
        remainder = len(all_rows) % pad_multiple
        if remainder:
            pad_count = pad_multiple - remainder
            pad_token_id = int(meta_info.get("pad_token_id", 0) or 0)
            for pad_idx in range(pad_count):
                all_rows.append({
                    "prompt_ids": [pad_token_id],
                    "response_ids": [],
                    "response_logprobs": None,
                    "response_mask": [],
                    "trajectory_reward": 0.0,
                    "training_score": 0.0,
                    "decision_loss_weight": 0.0,
                    "turn_id": -1,
                    "action_text": "",
                    "image_buffer_ref": None,
                    "image_indices": [],
                    "raw_prompt": "",
                    "mm_processor_kwargs": {},
                    "position_ids": [[0], [0], [0], [0]],
                    "uid": f"__vln_padding__{pad_idx}",
                    "trajectory_uid": f"__vln_padding__{pad_idx}",
                    "is_stop_action": False,
                    "is_padding": True,
                })

        n = len(all_rows)
        prompts_t = torch.zeros(n, prompt_length, dtype=torch.long)
        responses_t = torch.zeros(n, response_length, dtype=torch.long)
        attention_mask = torch.zeros(n, prompt_length + response_length, dtype=torch.long)
        response_mask = torch.zeros(n, response_length, dtype=torch.long)
        rm_scores = torch.zeros(n, response_length, dtype=torch.float32)
        position_ids = torch.zeros(n, 4, prompt_length + response_length, dtype=torch.long)

        for i, row in enumerate(all_rows):
            original_prompt_ids = row["prompt_ids"]
            original_response_ids = row["response_ids"]
            p_ids = original_prompt_ids[-prompt_length:]
            r_ids = original_response_ids[:response_length]
            r_mask = row["response_mask"][:response_length]

            pad_len = prompt_length - len(p_ids)
            prompts_t[i, pad_len:] = torch.tensor(p_ids, dtype=torch.long)
            attention_mask[i, pad_len:prompt_length] = 1

            responses_t[i, :len(r_ids)] = torch.tensor(r_ids, dtype=torch.long)
            attention_mask[i, prompt_length:prompt_length + len(r_ids)] = 1
            response_mask[i, :len(r_mask)] = torch.tensor(r_mask, dtype=torch.long)

            last_valid = len(r_ids) - 1
            if last_valid >= 0:
                rm_scores[i, last_valid] = float(row["training_score"])

            row_position_ids = row.get("position_ids")
            if row_position_ids is None:
                raise ValueError(
                    "VLN decision is missing processor-aligned 4-axis position_ids; "
                    f"trajectory_uid={row.get('trajectory_uid')!r}, turn_id={row.get('turn_id')!r}"
                )
            row_position_ids = torch.as_tensor(row_position_ids, dtype=torch.long)
            expected_length = len(original_prompt_ids) + len(original_response_ids)
            if row_position_ids.ndim != 2 or row_position_ids.shape != (4, expected_length):
                raise ValueError(
                    "VLN decision position_ids must have shape "
                    f"(4, prompt+response={expected_length}), got {tuple(row_position_ids.shape)}; "
                    f"trajectory_uid={row.get('trajectory_uid')!r}, turn_id={row.get('turn_id')!r}"
                )

            prompt_start = len(original_prompt_ids) - len(p_ids)
            position_ids[i, :, pad_len:prompt_length] = row_position_ids[
                :, prompt_start : len(original_prompt_ids)
            ]
            position_ids[i, :, prompt_length : prompt_length + len(r_ids)] = row_position_ids[
                :, len(original_prompt_ids) : len(original_prompt_ids) + len(r_ids)
            ]

        input_ids = torch.cat([prompts_t, responses_t], dim=1)

        non_tensor = {
            "uid": np.array([row["uid"] for row in all_rows], dtype=object),
            "trajectory_uid": np.array([row["trajectory_uid"] for row in all_rows], dtype=object),
            "trajectory_reward": np.array([row["trajectory_reward"] for row in all_rows], dtype=np.float32),
            "decision_loss_weight": np.array([row["decision_loss_weight"] for row in all_rows], dtype=np.float32),
            "turn_id": np.array([row["turn_id"] for row in all_rows], dtype=np.int32),
            "action_text": np.array([row["action_text"] for row in all_rows], dtype=object),
            "image_buffer_ref": np.array([row.get("image_buffer_ref") for row in all_rows], dtype=object),
            "image_indices": np.array([row.get("image_indices") for row in all_rows], dtype=object),
            "raw_prompt": np.array([row.get("raw_prompt") or "" for row in all_rows], dtype=object),
            "mm_processor_kwargs": np.array([row.get("mm_processor_kwargs") or {} for row in all_rows], dtype=object),
        }

        from tensordict import TensorDict
        batch = TensorDict({
            "prompts": prompts_t,
            "responses": responses_t,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
            "rm_scores": rm_scores,
        }, batch_size=n)

        output = DataProto(batch=batch, non_tensor_batch=non_tensor)
        output.meta_info = meta_info
        output.meta_info["vln_flattened"] = True
        output.meta_info["seqlen_sorted_indices"] = list(range(n))

        vln_metrics = dict(credit_metrics)
        if traj_rewards:
            r = np.array(traj_rewards)
            s = np.array(traj_successes)
            print(f"[VLN rollout] {len(traj_rewards)} trajectories, "
                  f"{real_decision_count} decisions (padded {n}), "
                  f"SR={s.mean():.1%}, reward={r.mean():.3f}±{r.std():.3f}")
            vln_metrics.update({
                "vln/traj_reward/mean": float(r.mean()),
                "vln/traj_reward/std": float(r.std()),
                "vln/traj_reward/max": float(r.max()),
                "vln/traj_reward/min": float(r.min()),
                "vln/traj_sr": float(s.mean()),
                "vln/traj_count": len(traj_rewards),
                "vln/avg_decisions_per_traj": real_decision_count / len(traj_rewards),
            })
        if vln_metrics:
            output.meta_info["vln_metrics"] = vln_metrics

        return output

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        window_size = self._get_rollout_window(len(prompts))
        total = len(prompts)
        scheduler = os.environ.get("VLN_ROLLOUT_SCHEDULER", "window").lower()

        if scheduler == "sliding":
            return await self._generate_sequences_sliding(prompts, window_size, total)

        if window_size >= total:
            # No windowing: original single-batch path
            one_to_one = await super().generate_sequences(prompts)
            rows, rewards, successes = self._extract_decisions(one_to_one)
            if not rows:
                return one_to_one
            return self._build_output(rows, rewards, successes, one_to_one.meta_info)

        # Fixed-window path: process rollouts in chunks
        all_rows = []
        all_rewards = []
        all_successes = []
        last_meta_info = None
        num_windows = (total + window_size - 1) // window_size

        for w_idx in range(num_windows):
            start = w_idx * window_size
            end = min(start + window_size, total)
            window = prompts[start:end]
            if hasattr(prompts, "meta_info"):
                window.meta_info = prompts.meta_info

            print(f"[VLN rollout] window {w_idx + 1}/{num_windows} "
                  f"({end - start} rollouts, range [{start}:{end}])")

            one_to_one = await super().generate_sequences(window)
            last_meta_info = one_to_one.meta_info

            rows, rewards, successes = self._extract_decisions(one_to_one)
            all_rows.extend(rows)
            all_rewards.extend(rewards)
            all_successes.extend(successes)

        if not all_rows:
            return one_to_one

        return self._build_output(all_rows, all_rewards, all_successes, last_meta_info)

    async def _generate_sequences_sliding(self, prompts, max_concurrent, total):
        """Bounded rolling queue: index queue + fixed worker loops, no barrier.

        Concurrency = min(VLN_ROLLOUT_WINDOW, num_workers, total).
        Workers pull episodes from a shared index queue; as each finishes,
        the next starts immediately — no window barrier.
        """
        concurrency = min(max_concurrent, len(self.agent_loop_workers), total)
        active_workers = self.agent_loop_workers[:concurrency]

        index_queue = asyncio.Queue()
        for i in range(total):
            index_queue.put_nowait(i)

        results = [None] * total
        all_metrics = []
        output_meta = {}
        completed = [0]

        async def worker_loop(worker):
            while True:
                try:
                    idx = index_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                single = prompts[idx:idx + 1]
                if hasattr(prompts, "meta_info"):
                    single.meta_info = prompts.meta_info

                out = await worker.generate_sequences.remote(single)
                rows, rewards, successes = self._extract_decisions(out)
                results[idx] = (rows, rewards, successes)

                worker_metrics = out.meta_info.get("metrics", [])
                all_metrics.extend(worker_metrics)
                for k, v in out.meta_info.items():
                    if k != "metrics" and k not in output_meta:
                        output_meta[k] = v

                completed[0] += 1
                if completed[0] % concurrency == 0 or completed[0] == total:
                    print(f"[VLN rollout] sliding: {completed[0]}/{total} episodes done")

        print(f"[VLN rollout] sliding mode: {total} episodes, concurrency={concurrency}")
        await asyncio.gather(*(worker_loop(w) for w in active_workers))

        all_rows, all_rewards, all_successes = [], [], []
        for res in results:
            if res:
                all_rows.extend(res[0])
                all_rewards.extend(res[1])
                all_successes.extend(res[2])

        if not all_rows:
            raise RuntimeError("[VLN rollout] sliding: all episodes returned empty decisions")

        meta_info = dict(prompts.meta_info) if hasattr(prompts, "meta_info") else {}
        meta_info.update(output_meta)
        output = self._build_output(all_rows, all_rewards, all_successes, meta_info)
        output.meta_info["timing"] = self._aggregate_timing(all_metrics)
        return output

    def _aggregate_timing(self, all_metrics):
        """Aggregate per-episode metrics into timing dict (mirrors AgentLoopManager._performance_metrics)."""
        timing = {}
        if not all_metrics:
            return timing

        flat = [m for chunk in all_metrics for m in (chunk if isinstance(chunk, list) else [chunk])]
        if not flat:
            return timing

        t_gen = np.array([m["generate_sequences"] for m in flat])
        t_tool = np.array([m["tool_calls"] for m in flat])
        t_score = np.array([m["compute_score"] for m in flat])
        num_preempted = np.array([m["num_preempted"] for m in flat])

        for prefix, arr in [
            ("agent_loop/generate_sequences", t_gen),
            ("agent_loop/tool_calls", t_tool),
            ("agent_loop/compute_score", t_score),
            ("agent_loop/num_preempted", num_preempted),
        ]:
            timing[f"{prefix}/min"] = float(arr.min())
            timing[f"{prefix}/max"] = float(arr.max())
            timing[f"{prefix}/mean"] = float(arr.mean())

        slowest = int(np.argmax(t_gen + t_tool + t_score))
        timing["agent_loop/slowest/generate_sequences"] = float(t_gen[slowest])
        timing["agent_loop/slowest/tool_calls"] = float(t_tool[slowest])
        timing["agent_loop/slowest/compute_score"] = float(t_score[slowest])
        timing["agent_loop/slowest/num_preempted"] = float(num_preempted[slowest])

        return timing
