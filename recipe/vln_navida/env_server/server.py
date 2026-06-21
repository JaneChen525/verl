"""FastAPI env server — session-based Habitat env pool (design doc §5-§7).

Each session = one Habitat episode on one dedicated worker process.
TTL background task reclaims stale sessions (worker crash / client forgot to delete).
"""
import logging
import multiprocessing as mp
import queue as _queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException

from recipe.vln_navida.env_server.schemas import (
    CreateSessionRequest, CreateSessionResponse,
    DeleteSessionResponse, EpisodeInfo, HealthResponse,
    Metrics, MetricsResponse, Observation,
    StatsResponse, StepRequest, StepResponse,
)
from recipe.vln_navida.env_server.worker import worker_loop

_START_TIME = time.time()
logger = logging.getLogger(__name__)


class WorkerCallTimeout(Exception):
    pass


# ── Worker handle ─────────────────────────────────────────────────────────────

class WorkerHandle:
    def __init__(self, worker_id: int, exp_config_path: str, gpu_id: int = -1,
                 split_override: str | None = None):
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.exp_config_path = exp_config_path
        self.split_override = split_override
        ctx = mp.get_context("spawn")
        self.cmd_q: mp.Queue = ctx.Queue()
        self.resp_q: mp.Queue = ctx.Queue()
        self.heartbeat = ctx.Value("d", 0.0)
        self.proc = ctx.Process(
            target=worker_loop,
            args=(worker_id, exp_config_path, self.cmd_q, self.resp_q, self.heartbeat, gpu_id,
                  split_override),
            daemon=True,
        )
        self.proc.start()
        ready = self.resp_q.get(timeout=300)
        if not ready.get("ok"):
            raise RuntimeError(f"worker {worker_id} failed: {ready.get('error')}")
        self.state: str = "idle"
        self.session_id: Optional[str] = None
        self.last_active: float = time.time()
        self._call_lock = threading.Lock()

    def call(self, cmd: dict, timeout: float = 60.0) -> dict:
        with self._call_lock:
            return self._call_inner(cmd, timeout)

    def _call_inner(self, cmd: dict, timeout: float) -> dict:
        self.last_active = time.time()
        request_id = uuid.uuid4().hex
        cmd = dict(cmd)
        cmd["request_id"] = request_id
        self.cmd_q.put(cmd)

        deadline = time.time() + timeout
        while True:
            remain = deadline - time.time()
            if remain <= 0:
                raise WorkerCallTimeout(
                    f"worker {self.worker_id} timeout on {cmd.get('op')}")
            try:
                r = self.resp_q.get(timeout=max(remain, 0.1))
            except _queue.Empty:
                raise WorkerCallTimeout(
                    f"worker {self.worker_id} timeout on {cmd.get('op')}")
            if r.get("request_id") == request_id:
                return r
            logger.warning("worker %d: dropped stale response (expected %s, got %s)",
                           self.worker_id, request_id, r.get("request_id"))

    def is_alive(self) -> bool:
        return self.proc.is_alive()

    def close(self):
        try:
            self.cmd_q.put({"op": "close"})
            self.proc.join(timeout=10)
        except Exception:
            pass
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(timeout=5)
        if self.proc.is_alive():
            self.proc.kill()


# ── Worker pool ───────────────────────────────────────────────────────────────

class WorkerPool:
    def __init__(self, exp_config_path: str, pool_size: int, session_ttl_sec: float,
                 gpu_ids: list[int] | None = None, split_override: str | None = None):
        self.pool_size = pool_size
        self.session_ttl_sec = session_ttl_sec
        self.exp_config_path = exp_config_path
        self.split_override = split_override
        if gpu_ids is None:
            gpu_ids = [-1] * pool_size
        elif len(gpu_ids) < pool_size:
            gpu_ids = gpu_ids * ((pool_size // len(gpu_ids)) + 1)
            gpu_ids = gpu_ids[:pool_size]
        self.gpu_ids = gpu_ids
        self.workers: list[WorkerHandle] = [
            WorkerHandle(i, exp_config_path, gpu_id=gpu_ids[i], split_override=split_override)
            for i in range(pool_size)
        ]
        self._session_to_worker: dict[str, WorkerHandle] = {}
        self._lock = threading.Lock()

    def claim(self, session_id: str) -> Optional[WorkerHandle]:
        with self._lock:
            for w in self.workers:
                if w.state == "idle" and w.is_alive():
                    w.state = "busy"
                    w.session_id = session_id
                    w.last_active = time.time()
                    self._session_to_worker[session_id] = w
                    return w
        return None

    def get(self, session_id: str) -> Optional[WorkerHandle]:
        return self._session_to_worker.get(session_id)

    def release(self, session_id: str):
        with self._lock:
            w = self._session_to_worker.pop(session_id, None)
            if w is not None:
                w.state = "idle"
                w.session_id = None

    def replace_worker(self, w: WorkerHandle):
        """Kill a broken worker and spawn a fresh one in its place."""
        worker_id = w.worker_id
        gpu_id = w.gpu_id
        logger.warning("replacing worker %d (gpu %d)", worker_id, gpu_id)

        with self._lock:
            if w.session_id is not None:
                self._session_to_worker.pop(w.session_id, None)
            w.state = "restarting"
            w.session_id = None

        w.close()

        new_w = WorkerHandle(
            worker_id, self.exp_config_path,
            gpu_id=gpu_id, split_override=self.split_override,
        )

        with self._lock:
            self.workers[worker_id] = new_w
        logger.info("worker %d replaced and ready", worker_id)

    def reap_stale(self):
        """Release sessions idle longer than TTL (worker crash or forgotten delete)."""
        now = time.time()
        with self._lock:
            stale = [sid for sid, w in self._session_to_worker.items()
                     if now - w.last_active > self.session_ttl_sec]
        for sid in stale:
            self.release(sid)

    @property
    def free_workers(self) -> int:
        return sum(1 for w in self.workers if w.state == "idle" and w.is_alive())

    @property
    def unhealthy_workers(self) -> int:
        return sum(1 for w in self.workers if not w.is_alive())

    def close(self):
        for w in self.workers:
            w.close()


# ── App state ─────────────────────────────────────────────────────────────────

_pool: Optional[WorkerPool] = None
_exp_config_path: Optional[str] = None
_pool_size: int = 1
_session_ttl_sec: float = 1800.0
_gpu_ids: list[int] | None = None
_split_override: Optional[str] = None


def _ttl_reaper():
    while True:
        time.sleep(30)
        if _pool is not None:
            _pool.reap_stale()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool
    assert _exp_config_path, "call set_config() before starting uvicorn"
    _pool = WorkerPool(_exp_config_path, _pool_size, _session_ttl_sec, gpu_ids=_gpu_ids,
                        split_override=_split_override)
    t = threading.Thread(target=_ttl_reaper, daemon=True)
    t.start()
    yield
    if _pool:
        _pool.close()


app = FastAPI(lifespan=lifespan)

_RESET_REQUIRED_KEYS = {"ok", "obs", "metrics", "scene_id", "episode_id", "instruction", "done"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _obs(d: dict) -> Observation:
    o = d["obs"]
    return Observation(rgb_jpeg_base64=o["rgb_jpeg_base64"],
                       width=o["width"], height=o["height"], step=o["step"])

def _metrics(d: dict) -> Metrics:
    m = d["metrics"]
    return Metrics(**m)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(ok=True)


@app.get("/stats", response_model=StatsResponse)
def stats():
    assert _pool
    return StatsResponse(
        pool_size=_pool.pool_size,
        free_workers=_pool.free_workers,
        active_sessions=len(_pool._session_to_worker),
        unhealthy_workers=_pool.unhealthy_workers,
        uptime_sec=time.time() - _START_TIME,
    )


@app.post("/v1/sessions", response_model=CreateSessionResponse)
def create_session(req: CreateSessionRequest):
    assert _pool
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    w = _pool.claim(session_id)
    if w is None:
        raise HTTPException(503, detail={"error": {"code": "NO_FREE_WORKER",
            "message": "No free Habitat worker", "retryable": True}})
    try:
        r = w.call({"op": "reset", "episode_id": req.episode_id, "session_id": session_id},
                   timeout=120)
    except WorkerCallTimeout as e:
        _pool.replace_worker(w)
        raise HTTPException(503, detail={"error": {"code": "RESET_TIMEOUT",
            "message": str(e), "retryable": True}})
    except Exception:
        _pool.release(session_id)
        raise

    if not r.get("ok"):
        err_msg = r.get("error", "")
        if "not found" in str(err_msg):
            _pool.release(session_id)
            raise HTTPException(400, detail=err_msg)
        _pool.replace_worker(w)
        raise HTTPException(503, detail={"error": {"code": "RESET_FAILED",
            "message": str(err_msg), "retryable": True}})

    missing = _RESET_REQUIRED_KEYS - set(r.keys())
    if missing:
        _pool.replace_worker(w)
        raise HTTPException(503, detail={"error": {"code": "BAD_WORKER_RESPONSE",
            "missing": sorted(missing), "retryable": True}})

    return CreateSessionResponse(
        session_id=session_id,
        worker_id=w.worker_id,
        episode=EpisodeInfo(scene_id=r["scene_id"], episode_id=r["episode_id"],
                            instruction=r["instruction"]),
        observation=_obs(r),
        metrics=_metrics(r),
        done=r["done"],
    )


@app.post("/v1/sessions/{session_id}/step", response_model=StepResponse)
def step(session_id: str, req: StepRequest):
    assert _pool
    w = _pool.get(session_id)
    if w is None:
        raise HTTPException(404, detail=f"session {session_id} not found")
    for a in req.actions:
        if a not in (0, 1, 2, 3):
            raise HTTPException(400, detail=f"invalid action {a}")
    try:
        r = w.call({"op": "step", "actions": req.actions}, timeout=60)
    except WorkerCallTimeout as e:
        _pool.replace_worker(w)
        raise HTTPException(503, detail={"error": {"code": "STEP_TIMEOUT",
            "message": str(e), "retryable": False}})
    if not r.get("ok"):
        raise HTTPException(500, detail=r.get("error"))
    if r["done"] and req.stop_on_done:
        _pool.release(session_id)
    return StepResponse(
        session_id=session_id,
        executed_actions=r["executed_actions"],
        observation=_obs(r),
        metrics=_metrics(r),
        done=r["done"],
        done_reason=r.get("done_reason"),
    )


@app.get("/v1/sessions/{session_id}/metrics", response_model=MetricsResponse)
def get_metrics(session_id: str):
    assert _pool
    w = _pool.get(session_id)
    if w is None:
        raise HTTPException(404, detail=f"session {session_id} not found")
    try:
        r = w.call({"op": "metrics"}, timeout=10)
    except WorkerCallTimeout as e:
        _pool.replace_worker(w)
        raise HTTPException(503, detail={"error": {"code": "METRICS_TIMEOUT",
            "message": str(e), "retryable": True}})
    if not r.get("ok"):
        raise HTTPException(500, detail=r.get("error"))
    return MetricsResponse(session_id=session_id, metrics=_metrics(r), done=r["done"])


@app.delete("/v1/sessions/{session_id}", response_model=DeleteSessionResponse)
def delete_session(session_id: str):
    assert _pool
    released = _pool.get(session_id) is not None
    _pool.release(session_id)
    return DeleteSessionResponse(session_id=session_id, released=released)


@app.get("/v1/episodes")
def list_episodes(limit: int = 200):
    """Convenience: return episode ids from the first worker's dataset."""
    assert _pool
    r = _pool.workers[0].call({"op": "episodes", "limit": limit}, timeout=15)
    if not r.get("ok"):
        raise HTTPException(500, detail=r.get("error"))
    return {"episode_ids": r["episode_ids"]}


def set_config(exp_config_path: str, pool_size: int = 1, session_ttl_sec: float = 1800.0,
               gpu_ids: list[int] | None = None, split_override: str | None = None):
    global _exp_config_path, _pool_size, _session_ttl_sec, _gpu_ids, _split_override
    _exp_config_path = exp_config_path
    _pool_size = pool_size
    _session_ttl_sec = session_ttl_sec
    _gpu_ids = gpu_ids
    _split_override = split_override
