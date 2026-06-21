"""Global Habitat env slot queue — bounds active rollouts to env_server pool size.

Ray named actor so all AgentLoopWorkerTQ processes share one queue.

Two gates:
  - rollout slots (capacity = pool_size): bounds total active rollouts
  - reset slots (reset_capacity): bounds concurrent resets to avoid
    overwhelming Habitat scene loading. Acquired before env.reset(),
    released after reset returns (before the step loop).

Usage in verl_agent_loop.py:
    queue = get_slot_queue()
    slot = await queue.acquire.remote(key)
    try:
        await queue.acquire_reset.remote(key)
        try:
            b64 = await env.reset(...)
        finally:
            await queue.release_reset.remote(key)
        # step loop ...
    finally:
        await queue.release.remote(key)
"""
import asyncio
import os

import ray


SLOT_QUEUE_VERSION = "v2"
SLOT_QUEUE_NAME = f"HabitatSlotQueue_{SLOT_QUEUE_VERSION}"
DEFAULT_CAPACITY = 32
DEFAULT_RESET_CAPACITY = 8


@ray.remote(num_cpus=0)
class HabitatSlotQueue:
    def __init__(self, capacity: int, reset_capacity: int):
        self.capacity = capacity
        self.reset_capacity = reset_capacity
        self._free: asyncio.Queue[int] = asyncio.Queue()
        for i in range(capacity):
            self._free.put_nowait(i)
        self._active: dict[str, int] = {}

        self._reset_free: asyncio.Queue[int] = asyncio.Queue()
        for i in range(reset_capacity):
            self._reset_free.put_nowait(i)
        self._reset_active: dict[str, int] = {}

    async def acquire(self, key: str) -> int:
        slot = await self._free.get()
        self._active[key] = slot
        return slot

    async def release(self, key: str) -> None:
        slot = self._active.pop(key, None)
        if slot is not None:
            self._free.put_nowait(slot)

    async def acquire_reset(self, key: str) -> int:
        slot = await self._reset_free.get()
        self._reset_active[key] = slot
        return slot

    async def release_reset(self, key: str) -> None:
        slot = self._reset_active.pop(key, None)
        if slot is not None:
            self._reset_free.put_nowait(slot)

    def stats(self) -> dict:
        return {
            "capacity": self.capacity,
            "active": len(self._active),
            "free": self._free.qsize(),
            "reset_capacity": self.reset_capacity,
            "reset_active": len(self._reset_active),
            "reset_free": self._reset_free.qsize(),
        }


def get_slot_queue() -> ray.actor.ActorHandle:
    capacity = int(os.environ.get("VLN_HABITAT_QUEUE_CAPACITY", DEFAULT_CAPACITY))
    reset_capacity = int(os.environ.get("VLN_HABITAT_RESET_CAPACITY", DEFAULT_RESET_CAPACITY))
    try:
        return ray.get_actor(SLOT_QUEUE_NAME)
    except ValueError:
        try:
            return HabitatSlotQueue.options(
                name=SLOT_QUEUE_NAME, lifetime="detached",
            ).remote(capacity, reset_capacity)
        except ValueError:
            return ray.get_actor(SLOT_QUEUE_NAME)
