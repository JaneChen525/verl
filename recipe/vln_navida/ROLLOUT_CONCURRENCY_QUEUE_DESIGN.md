# VLN Full-Episode Rollout Concurrency and Queue Design

本文设计 VLN full-episode online GRPO 在 P5 放大时的 rollout 并发控制机制，目标是允许 `train_batch_size * rollout.n` 远大于 env worker 数量，同时避免 vLLM V1 多模态缓存竞态和 env_server retry 风暴。

## 背景

当前 P5 目标可能达到：

```text
train_batch_size = 64
rollout.n = 8
total_rollouts_per_step = 512
```

每条 rollout 是一个完整 VLN episode：

```text
env.reset()
  -> 多轮:
       build NaVIDA prompt
       vLLM generate action
       env.step()
  -> trajectory reward
```

当 512 条 rollout 被一次性提交给 AgentLoopManager 时，系统会出现三类压力：

1. Habitat env worker 数量有限，例如 8 个。
2. vLLM 多模态图像处理并发过高，触发 V1 engine 的 mm cache TOCTOU 竞态。
3. Ray/driver 中同时存在大量 full-episode coroutine 和 trajectory data，内存与调度压力变大。

已观察到的典型错误：

```text
AssertionError: Expected a cached item for mm_hash='...'
vllm/multimodal/cache.py line 373
```

这个错误不是 env_server 崩溃，而是 vLLM V1 多模态 LRU cache 在并发请求下发生 TOCTOU：

```text
1. 请求 A 检查 cache，发现 mm_hash 已存在。
2. 请求 A 因为认为已缓存，于是不保留原始 image item。
3. 请求 B 触发 LRU 驱逐，删除该 mm_hash。
4. 请求 A merge cache 时再取 mm_hash，取不到 item。
5. item=None 且 cache miss，触发 assertion。
```

增大 `VLLM_MM_INPUT_CACHE_GIB` 可以降低驱逐概率，但不能消除该竞态。P5 需要显式控制 rollout 和 vLLM generate 并发。

## 设计目标

- 支持 `total_rollouts_per_step` 大于 env worker 数量。
- 支持 `total_rollouts_per_step` 大于 vLLM 可安全并发数。
- 保证 env worker 尽量不空转。
- 保证 vLLM generate 并发可控，降低或消除 mm cache 竞态。
- 保持 full-episode rollout 语义不变。
- 不依赖 vLLM 内部 LRU cache 正确性来保证训练稳定。
- 提供可分阶段放大的 P5 配置策略。

## 非目标

- 本设计不改变 GRPO advantage 算法。
- 本设计不改变 NaVIDA prompt 格式。
- 本设计不要求一次性把 512 条 rollout 全部并发执行。
- 本设计不把 env_server worker 数扩展为必须等于 rollout 数。
- 本设计不默认降低 vLLM 吞吐；全局 vLLM semaphore 是 `mm_hash` 复现后的 fallback，而不是 P5 起步默认项。

## 并发层次

需要把并发拆成三层独立控制：

```text
total_rollouts_per_step
  = train_batch_size * rollout.n

rollout_window
  = 同时被 VLN rollout manager 调度出来的 trajectory 数

env_slots
  = env_server 可同时运行的 Habitat session 数

vllm_generate_slots
  = 可选的全局 vLLM generate 限流；默认不启用，出现 mm_hash cache 竞态后再启用
```

示例：

```text
total_rollouts_per_step = 128
rollout_window = 32
env_slots = 8
vLLM max_num_seqs = 4
vllm_generate_slots = disabled
```

实际执行形态：

```text
128 total rollouts
  -> 每次只放 32 条进入 active window
  -> 其中最多 8 条拿到 env session
  -> 8 个 AgentLoopWorker 持续推进 episode
  -> vLLM 内部按 max_num_seqs=4 调度 generate
  -> 当前 window 完成后，再提交下一组 32 条
```

## 为什么 rollout_window 可以大于 env_workers

`rollout_window` 控制的是同时存在的 trajectory coroutine 数，不是同时占用 Habitat env 的数量。

如果：

```text
env_workers = 8
rollout_window = 16
```

则执行过程是：

```text
16 条 trajectory coroutine 同时存在
  -> 8 条成功创建 env session
  -> 8 条等待 env session
```

这通常是合理的，因为它可以让 env worker 一释放就马上被下一条 trajectory 接上，减少空转。

前提：

- 等待 env session 的 trajectory 不应占用 vLLM generate slot。
- env session 创建必须有可靠的等待机制。
- rollout_window 不应大到堆积大量 retry/coroutine。

当前 VLN 流程满足第一个前提，因为 `run_episode()` 是先 `env.reset()`，拿到第一帧后才调用 vLLM。

建议起步：

```text
env_workers = 8
rollout_window = 8   最稳，便于 debug
rollout_window = 12  轻微 overbook
rollout_window = 16  可用于吞吐优化
rollout_window >=32  不建议直接起步
```

## 总体架构

```text
VLNOnlineRolloutManager.generate_sequences()
  |
  |-- split repeated rollout batch into windows
  |
  |-- for each window:
  |     |
  |     |-- AgentLoopManager.generate_sequences(window)
  |     |     |
  |     |     |-- AgentLoopWorker
  |     |           |
  |     |           |-- VLNFullEpisodeAgentLoop.run()
  |     |                 |
  |     |                 |-- VLNEnv.reset()
  |     |                 |     -> env_server session queue
  |     |                 |
  |     |                 |-- for each decision:
  |     |                       |
  |     |                       |-- acquire global vLLM generate slot
  |     |                       |-- server_manager.generate()
  |     |                       |-- release global vLLM generate slot
  |     |                       |-- env.step()
  |     |
  |     |-- flatten trajectories in this window
  |     |-- append to output list
  |
  |-- concat flattened window outputs
  |-- return DataProto
```

## Component 1: rollout_window

### 作用

`rollout_window` 控制每次提交给 AgentLoopManager 的 rollout 数量。

没有 window 时：

```text
512 rollouts -> 一次性 submit -> 大量 coroutine + vLLM requests + env retry
```

有 window 时：

```text
512 rollouts -> 32 个 window，每个 16 条
```

### 推荐位置

在 `VLNOnlineRolloutManager.generate_sequences()` 中实现。

当前 manager 已经 override `generate_sequences()` 并负责 flatten trajectory，因此它是最适合加 window 的位置。

### 配置项

建议新增：

```yaml
actor_rollout_ref:
  rollout:
    agent:
      vln_rollout_window: 16
```

Hydra override：

```bash
++actor_rollout_ref.rollout.agent.vln_rollout_window=16
```

### 伪代码

```python
async def generate_sequences(self, prompts: DataProto) -> DataProto:
    window_size = self.rollout_config.agent.get("vln_rollout_window", len(prompts))
    outputs = []

    for start in range(0, len(prompts), window_size):
        window = prompts.slice(start, min(start + window_size, len(prompts)))
        one_to_one = await super().generate_sequences(window)
        flattened = self._flatten_trajectories(one_to_one)
        outputs.append(flattened)

    return DataProto.concat(outputs)
```

### 注意事项

- window concat 后仍要保持 batch tensor/non_tensor schema 一致。
- dummy padding 应在最终 concat 后统一做，或每个 window padding 后再在最终结果中过滤 dummy。
- 如果使用标准 GRPO，必须保证 `uid` grouping 不被 window 切坏。由于 repeated rollout batch 一般按 prompt interleave，window size 最好是 `rollout.n` 的倍数。

推荐约束：

```text
vln_rollout_window % rollout.n == 0
```

## Component 2: env_server session queue

### 当前机制

当前 `VLNEnv.reset()` 请求：

```text
POST /v1/sessions
```

当 env worker 全忙时，server 返回 503，client sleep 后重试。

这可以工作，但 P5 时会产生 retry 风暴：

```text
rollout_window=16, env_workers=8
  -> 8 条成功
  -> 8 条每 2s 重试
```

如果 rollout_window 更大，retry 更明显。

### 推荐机制

把 env_server 改成 server-side wait queue：

```text
POST /v1/sessions
  -> 如果有 idle worker，立即创建 session
  -> 如果没有 idle worker，进入 asyncio.Condition 等待
  -> worker release 后唤醒一个等待请求
  -> 超过 wait_timeout_s 返回 503/timeout
```

### API 建议

`CreateSessionRequest` 增加：

```python
wait_timeout_s: float = 300.0
queue_timeout_s: float | None = None
```

或直接复用：

```python
wait_timeout_s: float = 300.0
```

### 行为

```text
env_workers = 8
rollout_window = 16

前 8 条:
  claim worker -> reset -> session active

后 8 条:
  hold HTTP request in server queue
  不占 worker
  不占 vLLM slot
  不产生 retry storm
```

### TTL 与 heartbeat

如果 trajectory 在等待 vLLM generate 时长时间没有 env step，env session 可能被 TTL 回收。

建议：

- env client 在 episode 运行期间定期 heartbeat。
- env_server TTL 只回收没有 heartbeat 的 session。
- 等待队列中的 request 不创建 session，不需要 heartbeat。

## Component 3: optional global vLLM generate semaphore

### 问题

`actor_rollout_ref.rollout.max_num_seqs` 控制 vLLM 内部调度并发，但它不是 VLN 层面的全局调度器。

当多个 AgentLoopWorker 同时调用：

```python
server_manager.generate(...)
```

vLLM V1 内部多模态 cache 仍可能在并发请求中触发 mm_hash TOCTOU。

原先 8 GPU P4 能跑通时没有外层 `vllm_generate_slots`，实际并发控制来自：

```text
NUM_WORKERS = 8
actor_rollout_ref.rollout.max_num_seqs = 4
ROLLOUT_TP = 8  # 单 vLLM replica
```

因此 P5 起步不应直接加 `vllm_generate_slots=1/2`，否则可能过早牺牲 vLLM 吞吐。正确策略是：

```text
先保留原 8GPU 吞吐路径：
  NUM_WORKERS=8
  max_num_seqs=4
  no vllm_generate_slots

如果 mm_hash 复现：
  再启用 vllm_generate_slots 作为 fallback
```

### 目标

在需要时，在调用 vLLM 之前增加一个全局限流器：

```text
vllm_generate_slots = 1 / 2 / 4
```

启用后，任何 Ray worker 中的 `VLNFullEpisodeAgentLoop` 都必须先 acquire token，再调用 `server_manager.generate()`。

### 为什么不能用本地 asyncio.Semaphore

AgentLoopWorkers 分布在不同 Ray actor/process 中。

```python
asyncio.Semaphore(2)
```

只能限制单个进程内并发，不能限制全局 8 个 worker 的并发。

### 推荐实现：Ray named actor

新增一个轻量 Ray actor：

```python
@ray.remote
class AsyncLimiter:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.available = capacity
        self.waiters = deque()

    async def acquire(self, request_id: str):
        ...

    async def release(self, request_id: str):
        ...
```

初始化时创建 named actor：

```python
ray.get_actor("vln_vllm_generate_limiter")
```

如果不存在，则创建：

```python
AsyncLimiter.options(
    name="vln_vllm_generate_limiter",
    lifetime="detached",
).remote(capacity=vllm_generate_slots)
```

在 `VLNFullEpisodeAgentLoop.verl_decide()` 中：

```python
token = await limiter.acquire.remote(request_id)
try:
    output = await self.server_manager.generate(...)
finally:
    await limiter.release.remote(request_id)
```

### 启用策略

```text
default: disabled
  保留 vLLM max_num_seqs=4，优先保证吞吐

fallback-1: vllm_generate_slots = 4
  和 max_num_seqs=4 对齐，只做外层排队与可观测性

fallback-2: vllm_generate_slots = 2
  如果 slots=4 仍触发 mm_hash，降低多模态 generate 并发

fallback-3: vllm_generate_slots = 1
  最稳，用于确认是否完全消除 mm_hash assertion
```

### 超时与容错

必须支持：

- acquire timeout
- release finally
- request_id 去重
- actor 重启后的 fail-fast

建议：

```text
acquire_timeout_s = 600
generate_timeout_s = 300
```

如果 acquire timeout：

- 当前 trajectory 标记为 rollout failure。
- 释放 env session。
- 返回 reward=0 或跳过该 trajectory。

## Component 4: vLLM cache settings

无论是否启用全局 semaphore，都建议保留以下配置：

```bash
actor_rollout_ref.rollout.enable_prefix_caching=False
```

并保留环境变量：

```bash
VLLM_MM_INPUT_CACHE_GIB=8
```

如果当前 vLLM 支持，可尝试：

```bash
++actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True
```

如果不支持，再尝试：

```bash
++actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0
```

注意：这些参数与 vLLM 版本强相关，必须通过启动日志确认是否被接受。

## 推荐 P5 放大策略

不要从 `64 x 8` 直接起步。推荐阶梯如下。默认阶段不启用 `vllm_generate_slots`，先保留原 8GPU 的 vLLM 吞吐路径；只有在该阶段复现 `mm_hash` assertion 后，再启用 fallback 限流。

```text
P4 回归:
  train_batch_size=4
  rollout.n=4
  total_rollouts=16
  rollout_window=16
  env_workers=8
  max_num_seqs=4
  vllm_generate_slots=disabled

P5-1:
  train_batch_size=8
  rollout.n=4
  total_rollouts=32
  rollout_window=16
  env_workers=8
  max_num_seqs=4
  vllm_generate_slots=disabled

P5-2:
  train_batch_size=32
  rollout.n=4
  total_rollouts=128
  rollout_window=32
  env_workers=8
  max_num_seqs=4
  vllm_generate_slots=disabled

P5-3:
  train_batch_size=64
  rollout.n=4
  total_rollouts=256
  rollout_window=32 or 64
  env_workers=8 or 16
  max_num_seqs=4
  vllm_generate_slots=disabled first, fallback to 4/2/1 if mm_hash appears

P5-full:
  train_batch_size=64
  rollout.n=8
  total_rollouts=512
  rollout_window=32 or 64
  env_workers=8 or 16
  max_num_seqs=4
  vllm_generate_slots=disabled first, fallback to 4/2/1 if mm_hash appears
```

## 建议默认配置

8 GPU, env worker = 8，当前 `train_batch_size=32, rollout.n=4` 的合理起点：

```bash
TRAIN_BATCH_SIZE=32
ROLLOUT_N=4
NUM_WORKERS=8
TOTAL_STEPS=1

actor_rollout_ref.rollout.max_num_seqs=4
++actor_rollout_ref.rollout.agent.vln_rollout_window=32
actor_rollout_ref.rollout.enable_prefix_caching=False
```

如果仍触发 `mm_hash` assertion：

```bash
++actor_rollout_ref.rollout.agent.vln_vllm_generate_slots=4
```

如果 slots=4 仍触发：

```bash
++actor_rollout_ref.rollout.agent.vln_vllm_generate_slots=2
```

如果 slots=2 仍触发：

```bash
++actor_rollout_ref.rollout.agent.vln_vllm_generate_slots=1
```

如果 slots=1 仍触发：

```text
说明不是并发驱逐，而是 vLLM V1 multimodal cache 一致性 bug。
需要禁用 MM cache 或 patch vLLM cache fallback。
```

## 指标与观测

需要记录以下指标：

### rollout manager

```text
vln/rollout_window_size
vln/rollout_window_index
vln/rollout_window_duration_s
vln/rollouts_completed
vln/rollouts_failed
```

### env server

```text
env/free_workers
env/active_sessions
env/queued_session_requests
env/session_wait_time_p50
env/session_wait_time_p95
env/session_timeout_count
```

### vLLM limiter

```text
vllm_limiter/capacity
vllm_limiter/inflight
vllm_limiter/queued
vllm_limiter/wait_time_p50
vllm_limiter/wait_time_p95
vllm_limiter/acquire_timeout_count
```

### vLLM cache

如果 vLLM 暴露相关日志或 metric，记录：

```text
mm_cache_hit
mm_cache_miss
mm_cache_eviction
mm_hash_assertion_count
```

## Failure policy

### env session timeout

如果 trajectory 长时间拿不到 env session：

```text
释放所有已持有资源
记录 rollout failure
返回 reward=0 或跳过该 rollout
```

P5 初期建议 fail-fast，方便暴露瓶颈。

### vLLM acquire timeout

如果无法在 `acquire_timeout_s` 内拿到 generate token：

```text
关闭 env session
记录 vllm_limiter_timeout
trajectory reward=0
```

### vLLM generate exception

如果 `server_manager.generate()` 抛错：

```text
release vLLM token
close env session
记录 action_text=""
trajectory reward=0
```

不要让一个 generate exception 挂住整个 batch 的所有 slots。

## 实现顺序

### Phase 1: 配置级缓解

先不改架构，验证缓存相关开关：

```bash
trainer.val_before_train=False
actor_rollout_ref.rollout.enable_prefix_caching=False
VLLM_MM_INPUT_CACHE_GIB=8
```

如果支持：

```bash
++actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True
```

### Phase 2: rollout_window

在 `VLNOnlineRolloutManager` 中实现 windowed generate。

验收：

```text
total_rollouts=64
rollout_window=16
日志显示 4 个 window 顺序完成
env_server active_sessions <= env_workers
```

### Phase 3: global vLLM limiter

仅当默认路径复现 `mm_hash` assertion 时，实现 Ray named actor limiter，并在 `VLNFullEpisodeAgentLoop` 的 generate 前后 acquire/release。

验收：

```text
default disabled
P5-2 保持 NUM_WORKERS=8 + max_num_seqs=4 的吞吐路径

如果 mm_hash 复现：
  vllm_generate_slots=4 先验证外层排队是否足够
  vllm_generate_slots=2 再降低多模态 generate 并发
  vllm_generate_slots=1 最终兜底，确认是否完全消除竞态
```

### Phase 4: env_server wait queue

把 `503 + client retry` 改成 server-side wait queue。

验收：

```text
rollout_window=16
env_workers=8
无 503 retry storm
queued_session_requests 有短暂排队
active_sessions <= 8
```

## 最终建议

P5 的稳定路线：

```text
先 P4 小规模验证 checkpoint
然后启用 rollout_window，先不启用 vllm_generate_slots
保留 NUM_WORKERS=8 + max_num_seqs=4 的原 8GPU 吞吐路径
如果复现 mm_hash assertion，再按 4 -> 2 -> 1 启用 vllm_generate_slots
最后逐步放大 train_batch_size 和 rollout.n
```

不要依赖：

```text
只增大 VLLM_MM_INPUT_CACHE_GIB
只增加 env worker
只降低 max_num_seqs
```

这些可以降低概率，但不能提供可证明的 backpressure。最终完整机制是：

```text
rollout_window 控制 trajectory 总并发
env_server queue 控制 Habitat session 并发
global vLLM semaphore 作为 fallback 控制 multimodal generate 并发
```
