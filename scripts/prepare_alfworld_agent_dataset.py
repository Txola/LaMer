#!/usr/bin/env python3
"""Create local, auditable parquet inputs for an ALFWorld agent-loop run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from agent_system.environments.alfworld.envs import (
    get_environment,
    load_config_file,
)


def task_type(gamefile: str) -> str:
    trajectory_path = gamefile.replace("game.tw-pddl", "traj_data.json")
    with open(trajectory_path, encoding="utf-8") as stream:
        return json.load(stream)["task_type"]


def ordered_games(config_path: Path, eval_dataset: str, seed: int) -> list[str]:
    config = load_config_file(str(config_path))
    env_type = config["env"]["type"]
    environment = get_environment(env_type)(config, train_eval=eval_dataset)
    permutation = np.random.RandomState(seed).permutation(len(environment.game_files))
    return [environment.game_files[index] for index in permutation]


def make_rows(games: list[str], split: str) -> list[dict]:
    return [
        {
            "data_source": "alfworld",
            "prompt": [{"role": "user", "content": ""}],
            "ability": "agent",
            "agent_name": "alfworld_agent",
            "extra_info": {
                "split": split,
                "index": index,
                "gamefile": gamefile,
                "task_type": task_type(gamefile),
            },
        }
        for index, gamefile in enumerate(games)
    ]


def write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-dataset", default="eval_all")
    parser.add_argument("--validation-seed", type=int, default=1000)
    parser.add_argument("--validation-size", type=int, default=126)
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "agent_system/environments/alfworld/configs/config_tw.yaml"
        ),
    )
    args = parser.parse_args()

    games = ordered_games(args.config, args.eval_dataset, args.validation_seed)
    if args.validation_size <= 0 or args.validation_size > len(games):
        parser.error(
            f"--validation-size must be in [1, {len(games)}] for {args.eval_dataset}"
        )
    selected_games = games[: args.validation_size]

    # V1 VERL initializes both dataloaders even for validation-only runs.  The
    # train row is never rolled out when trainer.val_only=True.
    write_parquet(args.output_dir / "train.parquet", make_rows(selected_games[:1], "train"))
    write_parquet(args.output_dir / "test.parquet", make_rows(selected_games, "test"))

    manifest = {
        "eval_dataset": args.eval_dataset,
        "validation_seed": args.validation_seed,
        "validation_size": len(selected_games),
        "games": [
            {
                "index": index,
                "gamefile": gamefile,
                "task_type": task_type(gamefile),
            }
            for index, gamefile in enumerate(selected_games)
        ],
    }
    with open(args.output_dir / "manifest.json", "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")

    print(
        f"Wrote {len(selected_games)} {args.eval_dataset} validation tasks to "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
