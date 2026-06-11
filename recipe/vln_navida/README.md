# VLN NaVIDA GRPO Recipe

VLN（Vision-and-Language Navigation）全 episode 在线 GRPO 训练 recipe，基于 [verl](https://github.com/volcengine/verl) 框架。

模型在 Habitat 仿真环境中接收自然语言指令，根据 RGB 第一视角图像输出离散导航动作（stop / forward / turn left / turn right），通过 GRPO 优化导航策略。

## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│  verl-dev 容器 (8×A100 colocated: FSDP训练 + vLLM推理)       │
│                                                             │
│  run_grpo.sh → verl main_ppo                                │
│    ├── VLNOnlineRolloutManager (rollout_window 并发控制)      │
│    │     └── VLNFullEpisodeAgentLoop × num_workers           │
│    │           ├── apply_chat_template → vLLM generate       │
│    │           └── VLNEnv (HTTP) ──→ env_server              │
│    ├── Actor FSDP update (GRPO loss)                         │
│    └── Ref model logprob                                     │
└─────────────────────────────────────────────────────────────┘
                            │ HTTP
┌─────────────────────────────────────────────────────────────┐
│  Host (conda env: vln)                                      │
│  env_server (FastAPI, port 8002)                             │
│    └── WorkerPool: 8 × Habitat Env (每 GPU 1 worker)         │
│        R2R episodes → RGB渲染 → JPEG base64                  │
└─────────────────────────────────────────────────────────────┘
```

## 文件结构

```
recipe/vln_navida/
├── run_grpo.sh                 # 训练启动脚本（所有超参在此配置）
├── config/agent_loop.yaml      # Agent loop 注册配置
│
│  ── 训练核心 ──
├── online_rollout_manager.py   # VLNOnlineRolloutManager: rollout_window 并发控制 + 
│                               #   episode flatten（N traj → M decisions）
├── verl_agent_loop.py          # VLNFullEpisodeAgentLoop: 驱动单条完整 episode
├── full_episode_agent_loop.py  # run_episode(): episode 循环逻辑（decide→step→record）
├── vln_traj_grpo.py            # Trajectory-level GRPO advantage 说明
│
│  ── 数据 & Prompt ──
├── dataset.py                  # VLNEpisodeDataset: episode manifest → verl dataloader
├── make_manifest.py            # 生成 train/val parquet manifest
├── prompt.py                   # NaVIDA prompt 构建（b64 版 + verl marker 版）
├── action.py                   # 动作解析（extract_result + to_atomic_chunk）
├── reward.py                   # 轨迹 reward（success + optional progress shaping）
│
│  ── 环境服务 ──
├── env_pool.py                 # VLNEnv: 异步 HTTP client（env_server thin wrapper）
├── env_server/
│   ├── launch.py               # 启动入口（uvicorn）
│   ├── server.py               # FastAPI 服务 + WorkerPool + session 管理 + TTL 回收
│   ├── worker.py               # Habitat 渲染 worker（spawn 进程，per-GPU）
│   └── schemas.py              # Pydantic API schemas
│
│  ── 调试工具 ──
└── rollout_smoke.py            # P1 standalone rollout 测试（不依赖 verl）
```

## 前置条件

### 环境

| 组件 | 位置 | 说明 |
|------|------|------|
| Habitat + R2R 数据 | env1 host, conda `vln` | `data/vln_eval_datasets/r2r/`, MP3D 场景 |
| verl 0.8.0.dev | verl-dev 容器 | submodule `vln/reinforcement_learning/` |
| 模型权重 | 容器内 `/workspace/WorldModel/checkpoints/` | Qwen3VL_4B_R2R_RxR_swift |
| episode manifest | 容器内 `/root/data/` | `.parquet` 格式 |

### 生成 episode manifest

```bash
# env1 host, conda activate vln
cd /var/data0/sandbox/janec/WorldModel
python vln/reinforcement_learning/recipe/vln_navida/make_manifest.py \
    --data-dir data/vln_eval_datasets/r2r \
    --splits train val_unseen \
    --out-dir /root/data
# 输出: vln_r2r_train_10819.parquet, vln_r2r_val_unseen_1839.parquet
```

## 启动步骤

### Step 1: 启动 env_server（host 侧）

```bash
ssh root@10.117.29.198
conda activate vln
cd /var/data0/sandbox/janec/WorldModel

# 8 worker, 每 GPU 1 个, hfov=90, train split
PYTHONPATH=vln/reinforcement_learning:.:vln:$PYTHONPATH \
nohup python -m recipe.vln_navida.env_server.launch \
    --exp-config config/vln_r2r.yaml \
    --port 8002 \
    --pool-size 8 \
    --session-ttl-sec 120 \
    --gpu-ids 0,1,2,3,4,5,6,7 \
    --split train \
    > /tmp/env_server.log 2>&1 &

# 验证: 等待 "8 workers ready"
tail -f /tmp/env_server.log
```

> **注意**: `--split train` 加载训练集 episode；评测用 `--split val_unseen`。
> hfov **必须**与模型训练时一致（90°），否则 SR=0%。

### Step 2: 启动 GRPO 训练（容器内）

```bash
docker exec -it verl-dev bash
cd /workspace/WorldModel

# 默认配置: batch=32, n=4, 30 steps
TOTAL_STEPS=30 EXPERIMENT=my-experiment \
bash vln/reinforcement_learning/recipe/vln_navida/run_grpo.sh

# 或自定义参数:
TRAIN_BATCH_SIZE=16 ROLLOUT_N=8 TEMPERATURE=0.8 TOTAL_STEPS=100 \
EXPERIMENT=my-experiment \
bash vln/reinforcement_learning/recipe/vln_navida/run_grpo.sh
```

### Step 3: 监控

```bash
# 日志
docker exec verl-dev tail -f /workspace/WorldModel/logs/<experiment>.log

# wandb
# 自动上传到 wandb project "vln-grpo"
```

## 关键超参

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TRAIN_BATCH_SIZE` | 32 | 每 step 采样的 episode 数 |
| `ROLLOUT_N` | 4 | GRPO group size（每 episode 的 rollout 次数） |
| `VLN_ROLLOUT_WINDOW` / `ROLLOUT_WINDOW` | 8 | 并发窗口大小（每次提交给 vLLM 的 rollout 数） |
| `TEMPERATURE` | 0.6 | 采样温度（越高越多样） |
| `TOTAL_STEPS` | 339 | 训练步数（ceil(10819/32) = 1 epoch） |
| `SAVE_FREQ` | 10 | checkpoint 保存频率 |
| `ACTOR_LR` | 5e-7 | Actor 学习率 |
| `KL_LOSS_COEF` | 0.001 | KL 散度正则化系数 |
| `CLIP_GRAD` | 1.0 | 梯度裁剪 |
| `NUM_WORKERS` | 8 | Agent loop 并发 worker 数 |
| `GPU_MEM_UTIL` | 0.5 | vLLM GPU 显存利用率 |
| `VLLM_MM_INPUT_CACHE_GIB` | 8 | vLLM 多模态缓存大小 (GiB) |

实际每 step rollout 数 = `TRAIN_BATCH_SIZE × ROLLOUT_N`（默认 128）。

## 训练管线详解

### 每个 training step 的流程

```
1. verl 采样 batch_size=32 个 episode 起点
2. 每 episode 复制 n=4 → 128 个 rollout 任务
3. VLNOnlineRolloutManager 按 window=8 分批：
   for window in 128/8 = 16 个窗口:
     a. 8 个 VLNFullEpisodeAgentLoop 并发执行
     b. 每个 loop: reset env → (vLLM推理→解析动作→env.step) × 循环 → done
     c. 每次 vLLM 调用输出 ~2 个子动作 → ~4 个 atomic env step
     d. 平均 ~22 次模型调用 / 轨迹 ≈ ~88 个 env step
4. 128 条轨迹 flatten → ~2800 个 decision rows（每个 = 一次模型调用）
5. GRPO: 按 episode uid 分组，计算 group 内 advantage
6. Actor FSDP 更新（pg_loss + KL regularization）
7. 更新 vLLM 权重
```

### Reward 设计

当前为二值 reward：`success=1.0`（到达目标 3m 内）/ `0.0`。

```python
# reward.py
reward = success + progress_coef * oracle_success
```

`progress_coef` 默认 0.1（通过 `config/agent_loop.yaml` 配置）。

### rollout_window 并发控制

解决 vLLM V1 多模态缓存竞态问题（mm_hash TOCTOU race）。将 128 个 rollout 分成 16 个窗口顺序执行，每窗口 8 条并发。通过环境变量 `VLN_ROLLOUT_WINDOW` 控制。

## env_server API

| Endpoint | Method | 说明 |
|----------|--------|------|
| `POST /v1/sessions` | 创建 session（reset episode） | body: `{episode_id, config_path}` |
| `POST /v1/sessions/{id}/step` | 执行动作 | body: `{actions: [1,1,3], stop_on_done: true}` |
| `DELETE /v1/sessions/{id}` | 释放 session | — |
| `GET /v1/sessions/{id}/metrics` | 获取当前指标 | SR/SPL/NE/oracle_success |

Session 有 TTL 自动回收（默认 120s），防止 crash 后占满 worker pool。

## Checkpoint

保存路径: `checkpoints/vln-grpo/<experiment>/global_step_<N>/`

加载方式: verl 自动从上次 checkpoint 恢复（如果 `latest_checkpointed_iteration.txt` 存在）。

## 已知约束

- **hfov 必须匹配**：env_server 渲染的 hfov 必须 = 模型训练时的 hfov（90°），否则 SR=0%
- **TP=8 / dp=1**：vLLM 必须单 replica（TP=8），dp≥2 会触发 mm_hash cache bug
- **enable_prefix_caching=False**：缓解多模态缓存竞态
- **val_before_train=False**：env_server 只加载一个 split，初始验证会因找不到 episode 而 crash
- 容器内 `PYTHONPATH` 必须包含 `reinforcement_learning:WorldModel:WorldModel/vln`

## Troubleshooting

| 现象 | 原因 | 解决 |
|------|------|------|
| SR=0%, 模型从不 stop | hfov 不匹配 | 确认 `vln_r2r.yaml` 中 hfov=90 |
| 503 NO_FREE_WORKER | session 泄漏占满 pool | 重启 env_server（TTL 120s 后也会自动回收） |
| `AssertionError: Expected a cached item for mm_hash` | vLLM 多模态缓存竞态 | 确认 `enable_prefix_caching=False` + `VLLM_MM_INPUT_CACHE_GIB=8` + `VLN_ROLLOUT_WINDOW≤8` |
| `ConfigAttributeError` on agent config | Hydra 结构化 config 拒绝未知字段 | 用环境变量（如 `VLN_ROLLOUT_WINDOW`）替代 Hydra `++` |
| advantage=0, pg_loss=0 | 二值 reward 下 group 内 rollout 结果一致 | 增大 ROLLOUT_N / 提高 Temperature / 改用连续 reward |
| OOM during vLLM wake_up | gpu_mem_util 过高 | 降低 `GPU_MEM_UTIL`（≤0.5）或启用 param_offload |
