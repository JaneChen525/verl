#!/usr/bin/env bash
# VLN Full-Episode Online GRPO Training — V4 TQ path (report/025)
# Uses main_ppo_sync + AgentLoopManagerTQ + TransferQueue.
# Requires: env_server running on host (port 8002, hfov=90, --gpu-ids 0,...,7)
#
# Usage:
#   docker exec verl-dev bash -c 'cd /workspace/WorldModel && \
#     bash vln/reinforcement_learning/recipe/vln_navida/run_grpo_tq.sh'
#
# Override params:
#   TRAIN_BATCH_SIZE=8 ROLLOUT_N=4 TOTAL_STEPS=1 bash .../run_grpo_tq.sh

set -xeuo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
WORLDMODEL=${WORLDMODEL:-/workspace/WorldModel}
# Use submodule verl (has TQ/AgentLoopManagerTQ); pip verl uninstalled
export PYTHONPATH=${WORLDMODEL}/vln/reinforcement_learning:${WORLDMODEL}:${WORLDMODEL}/vln:${PYTHONPATH:-}

# Verify verl source
python3 -c "import verl; print('verl:', verl.__file__)"
python3 -c "import transfer_queue; print('TQ:', transfer_queue.__version__)"

# ── vLLM: disable custom all-reduce + symmetric memory (not supported on H100 NVL PCIe topology,
#    causes "CUDA driver error: operation not permitted" on 2nd TP replica with dp=2)
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
# ── Global Habitat env slot queue capacity. Bounds total active rollouts across
#    all workers to match env_server pool_size. Default 32 = 32 Habitat workers.
export VLN_HABITAT_QUEUE_CAPACITY=${VLN_HABITAT_QUEUE_CAPACITY:-32}
# ── Reset concurrency gate. Bounds concurrent Habitat scene resets to avoid
#    overwhelming GPU OpenGL rendering. Default 8 out of 32 active rollouts.
export VLN_HABITAT_RESET_CAPACITY=${VLN_HABITAT_RESET_CAPACITY:-8}
# ── vLLM multimodal cache
export VLLM_MM_INPUT_CACHE_GIB=${VLLM_MM_INPUT_CACHE_GIB:-8}
# ── Temp dirs: env3 root fs only 46GB, Ray session/spilling defaults to /tmp
export TMPDIR=${TMPDIR:-${WORLDMODEL}/../tmp}
export RAY_TMPDIR=${RAY_TMPDIR:-${TMPDIR}/ray}
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}"

MODEL_PATH=${MODEL_PATH:-${WORLDMODEL}/checkpoints/Qwen3VL_4B_R2R_RxR_swift}
TRAIN_FILE=${TRAIN_FILE:-/root/data/vln_r2r_train_10819.parquet}
VAL_FILE=${VAL_FILE:-/root/data/vln_r2r_val_unseen_1839.parquet}
AGENT_CFG=${AGENT_CFG:-${WORLDMODEL}/vln/reinforcement_learning/recipe/vln_navida/config/agent_loop.yaml}

# ── GPU / Parallelism ─────────────────────────────────────────────────────────
NGPUS=${NGPUS:-8}
ROLLOUT_TP=${ROLLOUT_TP:-8}
FSDP_SIZE=${FSDP_SIZE:-8}

# ── Rollout ───────────────────────────────────────────────────────────────────
ROLLOUT_N=${ROLLOUT_N:-4}             # GRPO group size
TEMPERATURE=${TEMPERATURE:-0.6}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.5}
NUM_WORKERS=${NUM_WORKERS:-8}
# No ROLLOUT_WINDOW — TQ path dispatches all prompts at once,
# concurrency bounded by num_workers + env_server pool_size

# ── Data ──────────────────────────────────────────────────────────────────────
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-256}

# ── Actor (PPO/GRPO update) ──────────────────────────────────────────────────
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-1}
ACTOR_LR=${ACTOR_LR:-5e-7}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
CLIP_GRAD=${CLIP_GRAD:-1.0}

# ── Logprob (old + ref) ──────────────────────────────────────────────────────
LOG_PROB_MICRO=${LOG_PROB_MICRO:-1}

# ── Trainer ───────────────────────────────────────────────────────────────────
TOTAL_STEPS=${TOTAL_STEPS:-339}
PROJECT=${PROJECT:-vln-grpo}
EXPERIMENT=${EXPERIMENT:-v4-tq-test}
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:--1}

# ── Launch (main_ppo_sync + TQ agent loop) ───────────────────────────────────
python3 -m verl.trainer.main_ppo_sync \
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
  +actor_rollout_ref.rollout.limit_images=9 \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.n=${ROLLOUT_N} \
  actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
  actor_rollout_ref.rollout.max_num_seqs=4 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.enable_prefix_caching=False \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO} \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_MODEL_LEN} \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_CFG}" \
  actor_rollout_ref.rollout.agent.default_agent_loop=vln_full_episode_agent_tq \
  actor_rollout_ref.rollout.agent.num_workers=${NUM_WORKERS} \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
  actor_rollout_ref.actor.optim.clip_grad=${CLIP_GRAD} \
  actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE} \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=${FSDP_SIZE} \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO} \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${MAX_MODEL_LEN} \
  trainer.logger='[console,wandb]' \
  trainer.project_name=${PROJECT} \
  trainer.experiment_name=${EXPERIMENT} \
  trainer.n_gpus_per_node=${NGPUS} \
  trainer.nnodes=1 \
  trainer.save_freq=${SAVE_FREQ} \
  trainer.test_freq=${TEST_FREQ} \
  trainer.val_before_train=False \
  trainer.total_epochs=1 \
  trainer.total_training_steps=${TOTAL_STEPS} \
  trainer.balance_batch=False \
  "$@"
