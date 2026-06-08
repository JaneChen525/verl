"""VLNEnv: per-trajectory Habitat env handle = thin async HTTP client over the
habitat env_server (design doc §12). Habitat runs in host conda; the verl
rollout worker only talks HTTP.

Session lifecycle: reset() -> step() x N -> close().
Images stay as raw env_server JPEG base64 (no re-encode) for byte-identical
pixels with eval_vllm_navida.py.
reset() retries 503 (no free worker) with a bound.
"""
import asyncio

import httpx


class VLNEnv:
    def __init__(self, base_url: str, config_path: str = "config/vln_r2r.yaml",
                 reset_timeout_s: float = 600.0, reset_retry_s: float = 2.0,
                 timeout: float = 120.0):
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout)
        self._config_path = config_path
        self._reset_timeout_s = reset_timeout_s
        self._reset_retry_s = reset_retry_s
        self.session_id: str | None = None
        self.instruction: str | None = None
        self._cur_b64: str | None = None
        self._last_metrics: dict | None = None
        self._done: bool = False

    async def reset(self, extra_info: dict) -> str:
        """Start one trajectory. Returns the first frame's JPEG base64."""
        body = {"episode_id": str(extra_info["episode_id"]),
                "config_path": extra_info.get("config_path", self._config_path)}
        deadline = asyncio.get_event_loop().time() + self._reset_timeout_s
        while True:
            try:
                r = await self._client.post("/v1/sessions", json=body)
            except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                if asyncio.get_event_loop().time() < deadline:
                    await asyncio.sleep(self._reset_retry_s)
                    continue
                raise
            if r.status_code == 503 and asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(self._reset_retry_s)
                continue
            r.raise_for_status()
            break
        d = r.json()
        self.session_id = d["session_id"]
        self.instruction = d["episode"]["instruction"]
        self._cur_b64 = d["observation"]["rgb_jpeg_base64"]
        self._last_metrics = d["metrics"]
        self._done = d["done"]
        return self._cur_b64

    async def step(self, actions: list[int]) -> dict:
        """Execute one or more atomic actions. Returns raw step response dict."""
        r = await self._client.post(
            f"/v1/sessions/{self.session_id}/step",
            json={"actions": actions, "stop_on_done": True},
        )
        r.raise_for_status()
        d = r.json()
        self._cur_b64 = d["observation"]["rgb_jpeg_base64"]
        self._last_metrics = d["metrics"]
        self._done = d["done"]
        return d

    def current_jpeg_b64(self) -> str:
        return self._cur_b64

    @property
    def done(self) -> bool:
        return self._done

    def metrics(self) -> dict:
        return self._last_metrics or {}

    async def close(self) -> None:
        try:
            if self.session_id is not None:
                await self._client.delete(f"/v1/sessions/{self.session_id}")
        finally:
            await self._client.aclose()
