"""Entrypoint: python -m recipe.vln_navida.env_server.launch --exp-config config/vln_r2r.yaml

Run on host (vln conda env, GPU 6):
  CUDA_VISIBLE_DEVICES=6 python -m recipe.vln_navida.env_server.launch \\
    --exp-config config/vln_r2r.yaml --port 8002 --pool-size 8 --session-ttl-sec 1800
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
    args = p.parse_args()
    set_config(args.exp_config, pool_size=args.pool_size, session_ttl_sec=args.session_ttl_sec)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
