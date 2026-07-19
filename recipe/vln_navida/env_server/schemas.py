"""Pydantic schemas for the VLN Habitat env server API (design doc §7)."""
from typing import Optional

from pydantic import BaseModel, Field


# ── Observation ──────────────────────────────────────────────────────────────

class Observation(BaseModel):
    rgb_jpeg_base64: str
    width: int
    height: int
    step: int


class Metrics(BaseModel):
    distance_to_goal: float
    success: float
    spl: float
    oracle_success: float
    oracle_navigation_error: float
    collisions: float
    position: list[float]
    heading: list[float]
    map_position: list[float] = Field(default_factory=list)


class EpisodeInfo(BaseModel):
    scene_id: str
    episode_id: str
    instruction: str


# ── POST /v1/sessions ─────────────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    episode_id: str
    config_path: str = "config/vln_r2r.yaml"
    request_id: Optional[str] = None


class CreateSessionResponse(BaseModel):
    session_id: str
    worker_id: int
    episode: EpisodeInfo
    observation: Observation
    metrics: Metrics
    done: bool
    done_reason: Optional[str] = None


# ── POST /v1/sessions/{session_id}/step ───────────────────────────────────────

class StepRequest(BaseModel):
    actions: list[int]          # list of atomic action ids: 0=stop 1=fwd 2=left 3=right
    stop_on_done: bool = True


class StepResponse(BaseModel):
    session_id: str
    executed_actions: list[int]
    observation: Observation
    metrics: Metrics
    done: bool
    done_reason: Optional[str] = None  # episode_over|stop_action|max_episode_steps|worker_error|timeout


# ── GET /v1/sessions/{session_id}/metrics ─────────────────────────────────────

class MetricsResponse(BaseModel):
    session_id: str
    metrics: Metrics
    done: bool


# ── DELETE /v1/sessions/{session_id} ─────────────────────────────────────────

class DeleteSessionResponse(BaseModel):
    session_id: str
    released: bool


# ── GET /health ───────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    ok: bool
    service: str = "vln-habitat-env-server"
    version: str = "0.1.0"


# ── GET /stats ────────────────────────────────────────────────────────────────

class StatsResponse(BaseModel):
    pool_size: int
    free_workers: int
    active_sessions: int
    unhealthy_workers: int
    uptime_sec: float
