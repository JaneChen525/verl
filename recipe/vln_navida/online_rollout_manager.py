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
import base64
import io

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
        import os
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
                    "turn_id": dec["turn_id"],
                    "action_text": dec.get("action_text", ""),
                    "is_stop_action": dec.get("is_stop_action", False),
                    "uid": group_uid,
                    "trajectory_uid": trajectory_uid,
                    "trajectory_reward": reward,
                    "decision_loss_weight": 1.0 / max(num_decisions, 1),
                })

        return all_rows, traj_rewards, traj_successes

    def _build_output(self, all_rows, traj_rewards, traj_successes, meta_info) -> DataProto:
        """Build padded DataProto from collected decision rows."""
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
            dummy = {k: all_rows[0][k] for k in all_rows[0]}
            dummy["response_mask"] = [0]
            dummy["trajectory_reward"] = 0.0
            dummy["image_buffer_ref"] = None
            dummy["image_indices"] = None
            dummy["raw_prompt"] = None
            for _ in range(pad_count):
                all_rows.append(dummy)

        n = len(all_rows)
        prompts_t = torch.zeros(n, prompt_length, dtype=torch.long)
        responses_t = torch.zeros(n, response_length, dtype=torch.long)
        attention_mask = torch.zeros(n, prompt_length + response_length, dtype=torch.long)
        response_mask = torch.zeros(n, response_length, dtype=torch.long)
        rm_scores = torch.zeros(n, response_length, dtype=torch.float32)

        for i, row in enumerate(all_rows):
            p_ids = row["prompt_ids"][-prompt_length:]
            r_ids = row["response_ids"][:response_length]
            r_mask = row["response_mask"][:response_length]

            pad_len = prompt_length - len(p_ids)
            prompts_t[i, pad_len:] = torch.tensor(p_ids, dtype=torch.long)
            attention_mask[i, pad_len:prompt_length] = 1

            responses_t[i, :len(r_ids)] = torch.tensor(r_ids, dtype=torch.long)
            attention_mask[i, prompt_length:prompt_length + len(r_ids)] = 1
            response_mask[i, :len(r_mask)] = torch.tensor(r_mask, dtype=torch.long)

            last_valid = len(r_ids) - 1
            if last_valid >= 0:
                rm_scores[i, last_valid] = float(row["trajectory_reward"])

        input_ids = torch.cat([prompts_t, responses_t], dim=1)

        position_ids = torch.zeros_like(input_ids)
        for i in range(n):
            non_pad = attention_mask[i].sum().item()
            pad = input_ids.shape[1] - non_pad
            position_ids[i, pad:] = torch.arange(non_pad)

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

        if traj_rewards:
            r = np.array(traj_rewards)
            s = np.array(traj_successes)
            print(f"[VLN rollout] {len(traj_rewards)} trajectories, "
                  f"{real_decision_count} decisions (padded {n}), "
                  f"SR={s.mean():.1%}, reward={r.mean():.3f}±{r.std():.3f}")
            output.meta_info["vln_metrics"] = {
                "vln/traj_reward/mean": float(r.mean()),
                "vln/traj_reward/std": float(r.std()),
                "vln/traj_reward/max": float(r.max()),
                "vln/traj_reward/min": float(r.min()),
                "vln/traj_sr": float(s.mean()),
                "vln/traj_count": len(traj_rewards),
                "vln/avg_decisions_per_traj": real_decision_count / len(traj_rewards),
            }

        return output

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        window_size = self._get_rollout_window(len(prompts))
        total = len(prompts)

        if window_size >= total:
            # No windowing: original single-batch path
            one_to_one = await super().generate_sequences(prompts)
            rows, rewards, successes = self._extract_decisions(one_to_one)
            if not rows:
                return one_to_one
            return self._build_output(rows, rewards, successes, one_to_one.meta_info)

        # Windowed path: process rollouts in chunks
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
