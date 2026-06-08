"""Entrypoint: python -m recipe.vln_navida.env_server.launch --exp-config config/vln_r2r.yaml

Run on host (vln conda env):
  python -m recipe.vln_navida.env_server.launch \
    --exp-config config/vln_r2r.yaml --port 8002 --pool-size 8 \
    --gpu-ids 0,1,2,3,4,5,6,7

Each worker gets its own GPU via CUDA_VISIBLE_DEVICES (1 worker per GPU).
If --gpu-ids has fewer entries than --pool-size, they are cycled round-robin.
"""
import argparse

import uvicorn

from recipe.vln_navida.env_server.server import app, set_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-config", required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8002)
    p.add_argument("--pool-size", type=int, default=1)
    p.add_argument("--session-ttl-sec", type=float, default=1800.0)
    p.add_argument("--gpu-ids", type=str, default=None,
                   help="Comma-separated GPU IDs, one per worker (e.g. 0,1,2,3,4,5,6,7)")
    p.add_argument("--split", type=str, default=None,
                   help="Override dataset split (e.g. train, val_unseen)")
    args = p.parse_args()
    gpu_ids = [int(g) for g in args.gpu_ids.split(",")] if args.gpu_ids else None
    set_config(args.exp_config, pool_size=args.pool_size,
               session_ttl_sec=args.session_ttl_sec, gpu_ids=gpu_ids,
               split_override=args.split)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
