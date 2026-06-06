#!/usr/bin/env bash
# VLN Full-Episode Online GRPO Training (design: report/016)
# Run inside verl-dev container on env1.
# Requires: env_server running on host (port 8002, hfov=90)
#
# Usage:
#   docker exec -it verl-dev bash
#   cd /workspace/WorldModel
#   bash vln/reinforcement_learning/recipe/vln_navida/run_grpo.sh
#
# Or override params:
#   TRAIN_BATCH_SIZE=4 ROLLOUT_N=4 TOTAL_STEPS=10 bash .../run_grpo.sh

set -xeuo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
WORLDMODEL=${WORLDMODEL:-/workspace/WorldModel}
export PYTHONPATH=${WORLDMODEL}/vln/reinforcement_learning:${WORLDMODEL}:${WORLDMODEL}/vln:${PYTHONPATH:-}

MODEL_PATH=${MODEL_PATH:-${WORLDMODEL}/checkpoints/Qwen3-VL-4B-vln-r2r-merged}
TRAIN_FILE=${TRAIN_FILE:-/root/data/vln_episodes_3.parquet}
VAL_FILE=${VAL_FILE:-${TRAIN_FILE}}
AGENT_CFG=${AGENT_CFG:-${WORLDMODEL}/vln/reinforcement_learning/recipe/vln_navida/config/agent_loop.yaml}

# ── GPU / Parallelism ─────────────────────────────────────────────────────────
# 4×A100-40GB colocated: FSDP(4) + vLLM TP(4) → dp_size = n_gpus/tp = 1
NGPUS=${NGPUS:-4}
ROLLOUT_TP=${ROLLOUT_TP:-4}          # vLLM tensor parallel
FSDP_SIZE=${FSDP_SIZE:-4}            # FSDP sharding across all GPUs

# ── Rollout ───────────────────────────────────────────────────────────────────
ROLLOUT_N=${ROLLOUT_N:-2}            # GRPO group size (trials per episode)
TEMPERATURE=${TEMPERATURE:-0.3}      # eval default; T>0 required for GRPO diversity
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096} # NaVIDA prompt ~2300 tok; 4096 fits without OOM
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.4}
NUM_WORKERS=${NUM_WORKERS:-4}        # agent loop workers; must <= batch*n and divide evenly

# ── Data ──────────────────────────────────────────────────────────────────────
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}  # episodes per training step
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-256}  # action text ~20-30 tokens

# ── Actor (PPO/GRPO update) ──────────────────────────────────────────────────
# KEY CONSTRAINTS (dp_size = NGPUS/TP = 1):
#   real_train_batch_size = TRAIN_BATCH_SIZE * ROLLOUT_N >= FSDP_SIZE
#   ppo_mini_batch_size <= real_train_batch_size
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-2}
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-1}  # per GPU
ACTOR_LR=${ACTOR_LR:-5e-7}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}

# ── Logprob (old + ref) ──────────────────────────────────────────────────────
LOG_PROB_MICRO=${LOG_PROB_MICRO:-1}

# ── Trainer ───────────────────────────────────────────────────────────────────
TOTAL_STEPS=${TOTAL_STEPS:-1}
PROJECT=${PROJECT:-vln-grpo}
EXPERIMENT=${EXPERIMENT:-qwen3vl-navida-full-episode}
SAVE_FREQ=${SAVE_FREQ:--1}
TEST_FREQ=${TEST_FREQ:--1}

# ── Launch ────────────────────────────────────────────────────────────────────
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
  data.max_prompt_length=${MAX_PROMPT_LENGTH} \
  data.max_response_length=${MAX_RESPONSE_LENGTH} \
  data.filter_overlong_prompts=False \
  data.truncation=error \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
  actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
  actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.n=${ROLLOUT_N} \
  actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
  actor_rollout_ref.rollout.max_num_seqs=4 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO} \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_MODEL_LEN} \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_CFG}" \
  actor_rollout_ref.rollout.agent.default_agent_loop=vln_full_episode_agent \
  actor_rollout_ref.rollout.agent.num_workers=${NUM_WORKERS} \
  ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=recipe.vln_navida.online_rollout_manager.VLNOnlineRolloutManager \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
  actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE} \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=${FSDP_SIZE} \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO} \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${MAX_MODEL_LEN} \
  trainer.logger='[console]' \
  trainer.project_name=${PROJECT} \
  trainer.experiment_name=${EXPERIMENT} \
  trainer.n_gpus_per_node=${NGPUS} \
  trainer.nnodes=1 \
  trainer.save_freq=${SAVE_FREQ} \
  trainer.test_freq=${TEST_FREQ} \
  trainer.total_epochs=1 \
  trainer.total_training_steps=${TOTAL_STEPS} \
  trainer.balance_batch=False \
  "$@"
