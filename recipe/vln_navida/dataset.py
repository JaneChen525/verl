"""VLNEpisodeDataset — episode manifest → verl dataloader dict (design doc §5, §8.1).

Each row is one resettable VLN episode start. The actual NaVIDA prompt is built
online at rollout time; raw_prompt is a placeholder required by verl's dataloader.
"""
import json
import pathlib
from typing import Union

from torch.utils.data import Dataset


class VLNEpisodeDataset(Dataset):
    """Loads a JSONL episode manifest.

    Each line:
      {"episode_id": "42", "scene_id": "17DRP5sb8fy",
       "instruction": "go down the hallway ...",   ← optional, informational
       "config_path": "config/vln_r2r.yaml"}        ← optional override
    """

    def __init__(self, data_files: Union[str, list[str]]):
        if isinstance(data_files, str):
            data_files = [data_files]
        self.rows: list[dict] = []
        for path in data_files:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        return {
            # verl dataloader expects raw_prompt; actual prompt is built online
            "raw_prompt": [{"role": "user", "content": "placeholder"}],
            "data_source": "vln",
            "agent_name": "vln_full_episode_agent",
            "extra_info": {
                "episode_id": str(row["episode_id"]),
                "scene_id": str(row.get("scene_id", "")),
                "instruction": str(row.get("instruction", "")),
                "config_path": str(row.get("config_path", "config/vln_r2r.yaml")),
            },
        }


def make_episode_manifest(episode_ids: list[str], config_path: str = "config/vln_r2r.yaml",
                          out_path: str = "/tmp/vln_episodes.jsonl") -> str:
    """Helper: write a simple manifest from a list of episode ids."""
    with open(out_path, "w") as f:
        for eid in episode_ids:
            f.write(json.dumps({"episode_id": eid, "config_path": config_path}) + "\n")
    return out_path
