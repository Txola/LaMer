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


def games_for_split(config_path: Path, dataset_split: str) -> list[str]:
    config = load_config_file(str(config_path))
    env_type = config["env"]["type"]
    environment = get_environment(env_type)(config, train_eval=dataset_split)
    return list(environment.game_files)


def ordered_games(config_path: Path, eval_dataset: str, seed: int) -> list[str]:
    games = games_for_split(config_path, eval_dataset)
    permutation = np.random.RandomState(seed).permutation(len(games))
    return [games[index] for index in permutation]


def make_rows(
    games: list[str], split: str, metadata: list[dict] | None = None
) -> list[dict]:
    if metadata is None:
        metadata = [{} for _ in games]
    if len(metadata) != len(games):
        raise ValueError("metadata must have one item per game")
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
                **metadata[index],
            },
        }
        for index, gamefile in enumerate(games)
    ]


def training_schedule(
    games: list[str], seed: int, batch_size: int, steps: int
) -> tuple[list[str], list[dict]]:
    """Reproduce the legacy launcher's independently shuffled task slots."""
    if batch_size <= 0:
        raise ValueError("training batch size must be positive")
    if steps <= 0:
        raise ValueError("training steps must be positive")

    streams = []
    for slot in range(batch_size):
        slot_games = list(games)
        rng = np.random.RandomState(seed + slot)

        def stream(slot_games=slot_games, rng=rng):
            while True:
                rng.shuffle(slot_games)
                yield from slot_games

        streams.append(stream())

    scheduled_games: list[str] = []
    metadata: list[dict] = []
    for step in range(steps):
        for slot, stream in enumerate(streams):
            scheduled_games.append(next(stream))
            metadata.append(
                {
                    "training_step": step,
                    "training_slot": slot,
                    "env_seed_offset": slot,
                }
            )
    return scheduled_games, metadata


def write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-dataset", default="eval_all")
    parser.add_argument("--validation-seed", type=int, default=1000)
    parser.add_argument("--validation-size", type=int, default=126)
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--training-steps", type=int, default=0)
    parser.add_argument("--train-batch-size", type=int, default=8)
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

    train_games = games_for_split(args.config, "train")
    if args.training_steps:
        scheduled_games, training_metadata = training_schedule(
            train_games,
            args.training_seed,
            args.train_batch_size,
            args.training_steps,
        )
    else:
        # V1 VERL initializes both dataloaders for validation-only runs.  Keep
        # that unused row valid in case it is inspected or rolled out manually.
        scheduled_games = train_games[:1]
        training_metadata = [
            {"training_step": 0, "training_slot": 0, "env_seed_offset": 0}
        ]

    write_parquet(
        args.output_dir / "train.parquet",
        make_rows(scheduled_games, "train", training_metadata),
    )
    write_parquet(args.output_dir / "test.parquet", make_rows(selected_games, "test"))

    manifest = {
        "eval_dataset": args.eval_dataset,
        "validation_seed": args.validation_seed,
        "validation_size": len(selected_games),
        "training_seed": args.training_seed,
        "training_steps": args.training_steps,
        "train_batch_size": args.train_batch_size,
        "training_rows": len(scheduled_games),
        "training_games": [
            {
                **training_metadata[index],
                "gamefile": gamefile,
                "task_type": task_type(gamefile),
            }
            for index, gamefile in enumerate(scheduled_games)
        ],
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
        f"Wrote {len(scheduled_games)} training rows and {len(selected_games)} "
        f"{args.eval_dataset} validation tasks to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
