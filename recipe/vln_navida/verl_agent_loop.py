"""VLN agent loops for verl (design doc §8.3).

Two variants:
  - VLNFullEpisodeAgentLoop (V3 legacy): returns single AgentLoopOutput with
    full trajectory in extra_fields. Used with VLNOnlineRolloutManager + main_ppo.
  - VLNFullEpisodeAgentLoopTQ (V4): returns list[AgentLoopOutput], one per
    decision. Used with AgentLoopManagerTQ + main_ppo_sync. Pre-computes
    multi_modal_inputs and position_ids in the agent loop (verl-core's
    decode→re-process breaks for VLN's multi-image prompts). verl-core handles
    reward broadcast, GRPO advantage, and TQ storage.
"""
import base64
import io
import os
from typing import Any
from uuid import uuid4

from PIL import Image

import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.tokenizer import build_multimodal_processor_inputs
from verl.workers.rollout.replica import TokenOutput

from recipe.vln_navida.env_pool import VLNEnv
from recipe.vln_navida.full_episode_agent_loop import (
    MAX_ACTION_HISTORY,
    DecisionGen,
    TrajectoryRecord,
    run_episode,
)
from recipe.vln_navida.prompt import build_navida_messages


def _b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


@register("vln_full_episode_agent")
class VLNFullEpisodeAgentLoop(AgentLoopBase):
    """One run() = one full episode. Decisions stored in extra_fields for flatten."""

    def __init__(self, *args, env_server_url: str = "http://127.0.0.1:8002",
                 progress_coef: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_server_url = os.environ.get("VLN_ENV_SERVER_URL", env_server_url)
        self.progress_coef = float(os.environ.get("VLN_PROGRESS_COEF", progress_coef))
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        extra = kwargs.get("extra_info", {}) or {}
        episode_id = str(extra.get("episode_id", ""))
        uid = str(kwargs.get("uid", episode_id))

        metrics: dict = {}

        # Build verl-native decide closure (uses server_manager + apply_chat_template)
        async def verl_decide(instruction: str, b64_buffer: list[str]) -> DecisionGen:
            pil_images = [_b64_to_pil(b) for b in b64_buffer]
            messages, images = build_navida_messages(instruction, pil_images)
            mm_processor_kwargs = self._get_mm_processor_kwargs(None)
            prompt_ids = await self.apply_chat_template(
                messages, images=images, mm_processor_kwargs=mm_processor_kwargs,
            )
            with simple_timer("generate_sequences", metrics):
                output: TokenOutput = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    image_data=images,
                    mm_processor_kwargs=mm_processor_kwargs,
                )
            action_text = self.tokenizer.decode(output.token_ids)
            return DecisionGen(
                action_text=action_text,
                prompt_ids=prompt_ids,
                response_ids=output.token_ids,
                response_logprobs=output.log_probs,
                response_mask=[1] * len(output.token_ids),
                images=images,
                mm_processor_kwargs=mm_processor_kwargs,
            )

        # Drive full episode
        env = VLNEnv(self.env_server_url)
        try:
            traj: TrajectoryRecord = await run_episode(
                env, extra, verl_decide,
                group_uid=uid,
                trajectory_uid=f"{uid}#{uuid4().hex[:8]}",
                progress_coef=self.progress_coef,
                max_env_steps=int(os.environ.get("VLN_MAX_ENV_STEPS", 400)),
            )
        finally:
            await env.close()

        # Pick last decision as the 1:1 placeholder output
        last = traj.decisions[-1] if traj.decisions else None
        if last and last.gen.prompt_ids is not None:
            prompt_ids = last.gen.prompt_ids
            response_ids = last.gen.response_ids or []
            response_mask = last.gen.response_mask or [1] * len(response_ids)
            response_logprobs = last.gen.response_logprobs
            multi_modal_data = {"images": last.gen.images} if last.gen.images else None
            mm_kwargs = last.gen.mm_processor_kwargs
        else:
            # fallback: empty (shouldn't happen)
            prompt_ids = [0]
            response_ids = [0]
            response_mask = [0]
            response_logprobs = None
            multi_modal_data = None
            mm_kwargs = None

        # Serialize full trajectory into extra_fields for the manager to flatten
        trajectory_data = {
            "group_uid": traj.group_uid,
            "trajectory_uid": traj.trajectory_uid,
            "episode_id": traj.episode_id,
            "scene_id": traj.scene_id,
            "instruction": traj.instruction,
            "reward": traj.reward,
            "metrics": traj.metrics,
            "num_decisions": len(traj.decisions),
            "decisions": [
                {
                    "turn_id": d.turn_id,
                    "prompt_ids": d.gen.prompt_ids,
                    "response_ids": d.gen.response_ids,
                    "response_logprobs": d.gen.response_logprobs,
                    "response_mask": d.gen.response_mask,
                    "images": d.gen.images,
                    "mm_processor_kwargs": d.gen.mm_processor_kwargs,
                    "action_text": d.action_text,
                    "is_stop_action": d.is_stop_action,
                }
                for d in traj.decisions
                if d.gen.prompt_ids is not None  # skip invalid decisions without token data
            ],
        }

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_kwargs,
            reward_score=float(traj.reward),
            num_turns=len(traj.decisions),
            metrics=metrics,
            extra_fields={"trajectory": trajectory_data},
        )


@register("vln_full_episode_agent_tq")
class VLNFullEpisodeAgentLoopTQ(AgentLoopBase):
    """V4 TQ version: run() returns list[AgentLoopOutput], one per decision.

    Pre-computes multi_modal_inputs and position_ids in the agent loop where we
    have the original images aligned with prompt_ids. Passes them via extra_fields
    (_precomputed_mm_inputs / _precomputed_position_ids) to bypass verl-core's
    decode→re-process which breaks for multi-image VLN prompts.
    verl-core handles: reward broadcast, GRPO advantage, TQ storage.
    """

    def __init__(self, *args, env_server_url: str = "http://127.0.0.1:8002",
                 progress_coef: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_server_url = os.environ.get("VLN_ENV_SERVER_URL", env_server_url)
        self.progress_coef = float(os.environ.get("VLN_PROGRESS_COEF", progress_coef))
        self.response_length = self.rollout_config.response_length

    def _precompute_mm_and_position(self, dec, r_ids_truncated):
        """Pre-compute multi_modal_inputs and position_ids for one decision
        using the original images (not decode→re-process)."""
        prompt_t = torch.tensor(dec.gen.prompt_ids, dtype=torch.long)
        response_t = torch.tensor(r_ids_truncated, dtype=torch.long)
        input_ids = torch.cat([prompt_t, response_t], dim=0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        mm_inputs = {}
        if dec.gen.images and self.processor is not None:
            mm_kwargs = dec.gen.mm_processor_kwargs or {}
            mm_result = build_multimodal_processor_inputs(
                self.processor,
                text=[self.tokenizer.decode(dec.gen.prompt_ids, skip_special_tokens=False)],
                images=dec.gen.images,
                mm_processor_kwargs=mm_kwargs,
            )
            mm_result.pop("input_ids", None)
            mm_result.pop("attention_mask", None)
            mm_inputs = dict(mm_result.convert_to_tensors("pt") if hasattr(mm_result, "convert_to_tensors") else mm_result)
            image_grid_thw = mm_inputs.get("image_grid_thw")
            if image_grid_thw is not None:
                mm_inputs["images_seqlens"] = torch.repeat_interleave(
                    image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]
                )

        position_ids = self._compute_position_ids(
            input_ids.unsqueeze(0), attention_mask.unsqueeze(0), mm_inputs
        ).squeeze(0)

        print(f"[DEBUG precompute] input_ids={input_ids.shape}, "
              f"mm_inputs keys={list(mm_inputs.keys())}, "
              f"image_grid_thw={mm_inputs.get('image_grid_thw', 'NONE')}, "
              f"position_ids={position_ids.shape}")

        return mm_inputs, position_ids

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> list[AgentLoopOutput]:
        extra = kwargs.get("extra_info", {}) or {}
        episode_id = str(extra.get("episode_id", ""))
        uid = str(kwargs.get("uid", episode_id))

        metrics: dict = {}

        async def verl_decide(instruction: str, b64_buffer: list[str]) -> DecisionGen:
            pil_images = [_b64_to_pil(b) for b in b64_buffer]
            messages, images = build_navida_messages(instruction, pil_images)
            mm_processor_kwargs = self._get_mm_processor_kwargs(None)
            prompt_ids = await self.apply_chat_template(
                messages, images=images, mm_processor_kwargs=mm_processor_kwargs,
            )
            with simple_timer("generate_sequences", metrics):
                output: TokenOutput = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    image_data=images,
                    mm_processor_kwargs=mm_processor_kwargs,
                )
            action_text = self.tokenizer.decode(output.token_ids)
            return DecisionGen(
                action_text=action_text,
                prompt_ids=prompt_ids,
                response_ids=output.token_ids,
                response_logprobs=output.log_probs,
                response_mask=[1] * len(output.token_ids),
                images=images,
                mm_processor_kwargs=mm_processor_kwargs,
            )

        env = VLNEnv(self.env_server_url)
        try:
            traj: TrajectoryRecord = await run_episode(
                env, extra, verl_decide,
                group_uid=uid,
                trajectory_uid=f"{uid}#{uuid4().hex[:8]}",
                progress_coef=self.progress_coef,
            )
        finally:
            await env.close()

        outputs: list[AgentLoopOutput] = []
        num_decisions = len(traj.decisions)

        for i, dec in enumerate(traj.decisions):
            if dec.gen.prompt_ids is None:
                continue
            is_final = (i == num_decisions - 1)
            r_ids = dec.gen.response_ids or []
            r_mask = dec.gen.response_mask or [1] * len(r_ids)
            r_logprobs = dec.gen.response_logprobs
            r_ids_trunc = r_ids[: self.response_length]

            mm_inputs, position_ids = self._precompute_mm_and_position(dec, r_ids_trunc)

            outputs.append(AgentLoopOutput(
                prompt_ids=dec.gen.prompt_ids,
                response_ids=r_ids_trunc,
                response_mask=r_mask[: self.response_length],
                response_logprobs=r_logprobs[: self.response_length] if r_logprobs else None,
                multi_modal_data=None,
                mm_processor_kwargs=dec.gen.mm_processor_kwargs,
                reward_score=float(traj.reward) if is_final else None,
                num_turns=num_decisions,
                metrics=metrics if is_final else {},
                extra_fields={
                    "trajectory_uid": traj.trajectory_uid,
                    "turn_id": dec.turn_id,
                    "action_text": dec.action_text,
                    "is_stop_action": dec.is_stop_action,
                    "_precomputed_mm_inputs": mm_inputs,
                    "_precomputed_position_ids": position_ids,
                },
            ))

        if not outputs:
            dummy_pos = torch.zeros(4, 2, dtype=torch.long)
            outputs.append(AgentLoopOutput(
                prompt_ids=[0],
                response_ids=[0],
                response_mask=[0],
                reward_score=0.0,
                num_turns=0,
                metrics=metrics,
                extra_fields={
                    "trajectory_uid": traj.trajectory_uid,
                    "empty": True,
                    "_precomputed_mm_inputs": {},
                    "_precomputed_position_ids": dummy_pos,
                },
            ))

        return outputs
