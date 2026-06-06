"""VLNOnlineRolloutManager — replaces AgentLoopManager for full-episode flatten (P2, design doc §8.4).

Standard flow:  1 row → 1 AgentLoopOutput → 1 training sample.
VLN flow:       1 row → 1 full episode → N decisions → N training samples (flatten).

This manager inherits AgentLoopManager and overrides generate_sequences:
1. Calls super() → 1:1 DataProto (each row = one episode-trial placeholder).
2. Extracts trajectory data (all decisions) from extra_fields["trajectory"].
3. Flattens: each decision → one padded row (prompt/response/mask/rm_scores/position_ids).
4. Returns the flattened DataProto (rows = total decisions across all trajectories).

The trainer receives a batch where each row is an INDEPENDENT NaVIDA decision, and
uid groups all decisions of the same episode start for trajectory-level GRPO (§10).
"""
import numpy as np
import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.protocol import DataProto
from verl.utils.ray_utils import auto_await


class VLNOnlineRolloutManager(AgentLoopManager):
    """Full-episode rollout + flatten decisions."""

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        # Step 1: standard 1:1 rollout (each row runs VLNFullEpisodeAgentLoop)
        one_to_one = await super().generate_sequences(prompts)

        # Step 2: extract trajectories from extra_fields + flatten
        prompt_length = self.rollout_config.prompt_length
        response_length = self.rollout_config.response_length

        trajectories = one_to_one.non_tensor_batch.get("trajectory")
        if trajectories is None:
            return one_to_one  # fallback: no trajectory data, return as-is

        # Collect all decisions from all trajectories
        all_rows = []
        for traj_data in trajectories:
            if traj_data is None:
                continue
            group_uid = traj_data["group_uid"]
            trajectory_uid = traj_data["trajectory_uid"]
            reward = traj_data["reward"]
            num_decisions = traj_data["num_decisions"]
            decisions = traj_data.get("decisions", [])

            for dec in decisions:
                all_rows.append({
                    "prompt_ids": dec["prompt_ids"],
                    "response_ids": dec["response_ids"],
                    "response_logprobs": dec.get("response_logprobs"),
                    "response_mask": dec["response_mask"],
                    "images": dec.get("images"),
                    "mm_processor_kwargs": dec.get("mm_processor_kwargs"),
                    "turn_id": dec["turn_id"],
                    "action_text": dec.get("action_text", ""),
                    "is_stop_action": dec.get("is_stop_action", False),
                    # trajectory-level fields (same for all decisions in this trajectory)
                    "uid": group_uid,           # GRPO groups by this
                    "trajectory_uid": trajectory_uid,
                    "trajectory_reward": reward,
                    "decision_loss_weight": 1.0 / max(num_decisions, 1),
                })

        if not all_rows:
            return one_to_one

        # Step 3: build padded tensors (manual left-pad prompt, right-pad response)
        n = len(all_rows)
        prompts_t = torch.zeros(n, prompt_length, dtype=torch.long)
        responses_t = torch.zeros(n, response_length, dtype=torch.long)
        attention_mask = torch.zeros(n, prompt_length + response_length, dtype=torch.long)
        response_mask = torch.zeros(n, response_length, dtype=torch.long)
        rm_scores = torch.zeros(n, response_length, dtype=torch.float32)

        for i, row in enumerate(all_rows):
            p_ids = row["prompt_ids"][-prompt_length:]   # truncate if too long
            r_ids = row["response_ids"][:response_length]
            r_mask = row["response_mask"][:response_length]

            # left-pad prompt
            pad_len = prompt_length - len(p_ids)
            prompts_t[i, pad_len:] = torch.tensor(p_ids, dtype=torch.long)
            attention_mask[i, pad_len:prompt_length] = 1

            # right-pad response
            responses_t[i, :len(r_ids)] = torch.tensor(r_ids, dtype=torch.long)
            attention_mask[i, prompt_length:prompt_length + len(r_ids)] = 1
            response_mask[i, :len(r_mask)] = torch.tensor(r_mask, dtype=torch.long)

            # rm_scores: trajectory reward at last valid response token
            last_valid = len(r_ids) - 1
            if last_valid >= 0:
                rm_scores[i, last_valid] = float(row["trajectory_reward"])

        input_ids = torch.cat([prompts_t, responses_t], dim=1)

        # position_ids: sequential for non-pad tokens (simplified; mrope handled by verl's compute_position_ids downstream)
        position_ids = torch.zeros_like(input_ids)
        for i in range(n):
            non_pad = attention_mask[i].sum().item()
            pad = input_ids.shape[1] - non_pad
            position_ids[i, pad:] = torch.arange(non_pad)

        # Step 4: build non-tensor batch
        non_tensor = {
            "uid": np.array([row["uid"] for row in all_rows], dtype=object),
            "trajectory_uid": np.array([row["trajectory_uid"] for row in all_rows], dtype=object),
            "trajectory_reward": np.array([row["trajectory_reward"] for row in all_rows], dtype=np.float32),
            "decision_loss_weight": np.array([row["decision_loss_weight"] for row in all_rows], dtype=np.float32),
            "turn_id": np.array([row["turn_id"] for row in all_rows], dtype=np.int32),
            "action_text": np.array([row["action_text"] for row in all_rows], dtype=object),
        }

        # Step 5: assemble DataProto
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
        output.meta_info = one_to_one.meta_info  # carry timing etc.
        return output
