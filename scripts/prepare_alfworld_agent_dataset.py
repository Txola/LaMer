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


ALFWORLD_TASK_TYPES = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
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


def balanced_game_order(games: list[str], seed: int) -> list[str]:
    """Shuffle within task types, then interleave all six task types."""
    buckets = {task: [] for task in ALFWORLD_TASK_TYPES}
    for game in games:
        game_task_type = task_type(game)
        if game_task_type not in buckets:
            raise ValueError(f"Unexpected ALFWorld task type: {game_task_type}")
        buckets[game_task_type].append(game)

    missing = [task for task, task_games in buckets.items() if not task_games]
    if missing:
        raise ValueError(f"ALFWorld split is missing task types: {', '.join(missing)}")

    rng = np.random.RandomState(seed)
    task_order = [ALFWORLD_TASK_TYPES[index] for index in rng.permutation(6)]
    for task_games in buckets.values():
        rng.shuffle(task_games)

    ordered: list[str] = []
    offsets = {task: 0 for task in ALFWORLD_TASK_TYPES}
    while len(ordered) < len(games):
        for task in task_order:
            offset = offsets[task]
            if offset < len(buckets[task]):
                ordered.append(buckets[task][offset])
                offsets[task] += 1
    return ordered


def ordered_games(config_path: Path, eval_dataset: str, seed: int) -> list[str]:
    return balanced_game_order(games_for_split(config_path, eval_dataset), seed)


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
    """Build a deterministic schedule balanced across the six task types."""
    if batch_size <= 0:
        raise ValueError("training batch size must be positive")
    if steps <= 0:
        raise ValueError("training steps must be positive")

    scheduled_games: list[str] = []
    epoch = 0
    target_size = batch_size * steps
    while len(scheduled_games) < target_size:
        epoch_games = balanced_game_order(games, seed + epoch)
        remaining = target_size - len(scheduled_games)
        scheduled_games.extend(epoch_games[:remaining])
        epoch += 1

    metadata = [
        {
            "training_step": index // batch_size,
            "training_slot": index % batch_size,
            "env_seed_offset": index % batch_size,
        }
        for index in range(target_size)
    ]
    return scheduled_games, metadata


def task_counts(games: list[str]) -> dict[str, int]:
    counts = {task: 0 for task in ALFWORLD_TASK_TYPES}
    for game in games:
        counts[task_type(game)] += 1
    return counts


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
        "training_task_counts": task_counts(scheduled_games),
        "validation_task_counts": task_counts(selected_games),
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
