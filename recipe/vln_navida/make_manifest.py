"""Generate episode manifest parquet files for verl training.

Usage (on env1):
    conda activate vln
    cd /var/data0/sandbox/janec/WorldModel
    python vln/reinforcement_learning/recipe/vln_navida/make_manifest.py \
        --data-dir data/vln_eval_datasets/r2r \
        --splits train val_unseen \
        --out-dir /root/data
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_episodes(data_dir: Path, split: str) -> list[dict]:
    path = data_dir / split / f"{split}.json.gz"
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    return data["episodes"]


def episodes_to_parquet(episodes: list[dict], out_path: Path):
    rows = []
    for ep in episodes:
        rows.append({
            "data_source": "vln",
            "prompt": [{"role": "user", "content": "placeholder"}],
            "agent_name": "vln_full_episode_agent_tq",
            "extra_info": {
                "episode_id": str(ep["episode_id"]),
                "scene_id": str(ep.get("scene_id", "")),
                "instruction": str(ep.get("instruction", {}).get("instruction_text", ""))
                    if isinstance(ep.get("instruction"), dict)
                    else str(ep.get("instruction", "")),
                "config_path": "config/vln_r2r.yaml",
            },
        })
    df = pd.DataFrame(rows)
    df["prompt"] = df["prompt"].apply(lambda x: np.array(x, dtype=object))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"  {out_path}: {len(df)} episodes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/vln_eval_datasets/r2r"))
    parser.add_argument("--splits", nargs="+", default=["train", "val_unseen"])
    parser.add_argument("--out-dir", type=Path, default=Path("/root/data"))
    args = parser.parse_args()

    for split in args.splits:
        episodes = load_episodes(args.data_dir, split)
        n = len(episodes)
        out_name = f"vln_r2r_{split}_{n}.parquet"
        episodes_to_parquet(episodes, args.out_dir / out_name)


if __name__ == "__main__":
    main()
