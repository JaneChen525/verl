"""P1 rollout smoke: drive full episodes via env_server + vLLM OpenAI API, save trace.

Validates env_server session API + NaVIDA prompt + action parser + run_episode
against eval_vllm_navida.py before wiring into verl (P2).

Launch env_server (host):
  CUDA_VISIBLE_DEVICES=6 python -m recipe.vln_navida.env_server.launch \\
    --exp-config config/vln_r2r.yaml --port 8002 --pool-size 8

Launch vLLM (container):
  CUDA_VISIBLE_DEVICES=0 vllm serve <model> --port 8000 --max-model-len 8192 ...

Run:
  cd <VLN repo>
  PYTHONPATH=vln/reinforcement_learning python3 -m recipe.vln_navida.rollout_smoke \\
    --env-url http://127.0.0.1:8002 --vllm-url http://127.0.0.1:8000 \\
    --model <served-name> --n-episodes 8 --concurrency 8 --out /tmp/trace.json
"""
import argparse
import asyncio
import json

import httpx

from recipe.vln_navida.env_pool import VLNEnv
from recipe.vln_navida.full_episode_agent_loop import DecisionGen, run_episode
from recipe.vln_navida.prompt import build_navida_messages_b64


def make_decide(client: httpx.AsyncClient, model: str, temperature: float,
                top_p: float, max_tokens: int):
    async def decide(instruction: str, b64_buffer: list) -> DecisionGen:
        messages = build_navida_messages_b64(instruction, b64_buffer)
        r = await client.post("/v1/chat/completions", json={
            "model": model, "messages": messages,
            "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens,
        })
        r.raise_for_status()
        return DecisionGen(action_text=r.json()["choices"][0]["message"]["content"])
    return decide


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-url", default="http://127.0.0.1:8002")
    ap.add_argument("--vllm-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--config-path", default="config/vln_r2r.yaml")
    ap.add_argument("--n-episodes", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-decisions", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", default="/tmp/vln_trace.json")
    args = ap.parse_args()

    # fetch episode ids from env_server (GET /v1/episodes) or --episode-ids / --manifest
    episode_ids: list[str] = []
    if hasattr(args, 'episode_ids') and args.episode_ids:
        episode_ids = [e.strip() for e in args.episode_ids.split(",") if e.strip()]
    async with httpx.AsyncClient(base_url=args.env_url, timeout=30) as ec:
        if not episode_ids:
            r = await ec.get("/v1/episodes", params={"limit": args.n_episodes})
            r.raise_for_status()
            episode_ids = r.json()["episode_ids"]
    episode_ids = episode_ids[: args.n_episodes]

    vclient = httpx.AsyncClient(base_url=args.vllm_url, timeout=120.0)
    decide = make_decide(vclient, args.model, args.temperature, args.top_p, args.max_tokens)
    sem = asyncio.Semaphore(args.concurrency)

    async def one(ep: str):
        async with sem:
            env = VLNEnv(args.env_url, config_path=args.config_path)
            try:
                traj = await run_episode(
                    env, {"episode_id": ep, "config_path": args.config_path},
                    decide, group_uid=ep, trajectory_uid=f"{ep}#0",
                    max_decisions=args.max_decisions,
                )
            finally:
                await env.close()
            print(f"ep={ep} SR={traj.reward} decisions={traj.metrics['num_decisions']} "
                  f"steps={traj.metrics['env_steps']} ne={traj.metrics.get('distance_to_goal',0):.2f}",
                  flush=True)
            return traj

    trajs = await asyncio.gather(*[one(ep) for ep in episode_ids])
    await vclient.aclose()

    sr = sum(t.reward for t in trajs) / max(len(trajs), 1)
    print(f"\n=== {len(trajs)} episodes | SR={sr:.3f} ===")

    with open(args.out, "w") as f:
        json.dump([{
            "episode_id": t.episode_id, "reward": t.reward, "metrics": t.metrics,
            "decisions": [{"turn_id": d.turn_id, "action_text": d.action_text,
                           "parsed": d.parsed_actions, "atomic": d.atomic_chunk}
                          for d in t.decisions],
        } for t in trajs], f, indent=2)
    print(f"trace -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
