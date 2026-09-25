#!/usr/bin/env python3
"""Build LaMer's task-category ALFWorld splits from the official download.

The paper treats Pick, Look, Clean, and Heat as in-distribution task types and
Cool and Pick2 as out-of-distribution task types. Validation combines
ALFWorld's ``valid_train`` and ``valid_seen`` splits. Both contain held-out task
instances in training-distribution rooms, so task category remains the intended
distribution shift rather than room-scene novelty.

In addition to the complete validation pools, the script creates fixed,
task-balanced checkpoint splits containing the same sampled games used by the
all-task evaluation: one for the four in-distribution task types and one for
all six task types. The generated tree contains real directories and symlinks
to the original trial files. The source download is never changed or
duplicated.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import random
from collections import Counter
from pathlib import Path
import shutil
import tempfile


ID_TASK_TYPES = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
)
OOD_TASK_TYPES = (
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
)
ALL_TASK_TYPES = ID_TASK_TYPES + OOD_TASK_TYPES


@dataclass(frozen=True)
class Trial:
    source_split: Path
    path: Path
    task_type: str

    @property
    def relative_path(self) -> Path:
        return self.path.relative_to(self.source_split)

    @property
    def manifest_path(self) -> str:
        return str(Path(self.source_split.name) / self.relative_path)


def default_data_root() -> Path:
    return Path(os.environ.get("ALFWORLD_DATA", "~/.cache/alfworld")).expanduser()


def eligible_trials(split_root: Path, allowed: set[str]) -> tuple[list[Trial], Counter]:
    trials: list[Trial] = []
    counts: Counter = Counter()

    for metadata_path in sorted(split_root.rglob("traj_data.json")):
        trial_dir = metadata_path.parent
        relative = trial_dir.relative_to(split_root)
        relative_text = str(relative)

        # Match the filtering performed by AlfredTWEnv.collect_game_files.
        if "movable" in relative_text or "Sliced" in relative_text:
            counts["skipped_unsupported_path"] += 1
            continue

        with metadata_path.open() as stream:
            metadata = json.load(stream)
        task_type = metadata.get("task_type")
        if task_type not in allowed:
            counts["skipped_task_type"] += 1
            continue

        game_path = trial_dir / "game.tw-pddl"
        if not game_path.is_file():
            counts["skipped_missing_game"] += 1
            continue

        with game_path.open() as stream:
            game = json.load(stream)
        if not game.get("solvable", False):
            counts["skipped_unsolvable"] += 1
            continue

        trials.append(Trial(split_root, trial_dir, task_type))
        counts[task_type] += 1

    return trials, counts


def collect_trials(source_splits: tuple[Path, ...], allowed: set[str]) -> list[Trial]:
    trials: list[Trial] = []
    for source_split in source_splits:
        source_trials, _ = eligible_trials(source_split, allowed)
        trials.extend(source_trials)
    return trials


def task_counts(trials: list[Trial]) -> dict[str, int]:
    counts = Counter(trial.task_type for trial in trials)
    return {task: counts[task] for task in ALL_TASK_TYPES if counts[task]}


def link_trial(trial: Trial, destination_split: Path, include_source: bool) -> None:
    relative = trial.relative_path
    if include_source:
        relative = Path(trial.source_split.name) / relative
    destination_trial = destination_split / relative
    destination_trial.mkdir(parents=True, exist_ok=False)

    for source_path in trial.path.rglob("*"):
        relative = source_path.relative_to(trial.path)
        destination_path = destination_trial / relative
        if source_path.is_dir():
            destination_path.mkdir(parents=True, exist_ok=True)
        elif source_path.is_file():
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            destination_path.symlink_to(source_path.resolve())


def parse_args() -> argparse.Namespace:
    data_root = default_data_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=data_root / "json_2.1.1",
        help="Official ALFWorld JSON root (default: %(default)s)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=data_root / "json",
        help="LaMer split output root (default: %(default)s)",
    )
    parser.add_argument(
        "--validation-splits",
        nargs="+",
        choices=("valid_seen", "valid_unseen", "valid_train"),
        default=("valid_train", "valid_seen"),
        help="Official splits combined to form validation pools",
    )
    parser.add_argument(
        "--checkpoint-games-per-task",
        type=int,
        default=21,
        help="Number of fixed checkpoint-evaluation games per task type",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Random seed used to select the fixed balanced subset",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print counts without creating the output tree",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Atomically replace an existing generated output tree",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    train_root = source_root / "train"
    validation_roots = tuple(source_root / name for name in args.validation_splits)

    for required in (train_root, *validation_roots):
        if not required.is_dir():
            raise SystemExit(f"Missing ALFWorld source split: {required}")

    definitions = {
        "train_first4": ((train_root,), set(ID_TASK_TYPES)),
        "valid_first4": (validation_roots, set(ID_TASK_TYPES)),
        "valid_last2": (validation_roots, set(OOD_TASK_TYPES)),
        "valid_all": (validation_roots, set(ALL_TASK_TYPES)),
    }

    selected: dict[str, list[Trial]] = {}
    for name, (source_splits, allowed) in definitions.items():
        trials = collect_trials(source_splits, allowed)
        if not trials:
            sources = ", ".join(map(str, source_splits))
            raise SystemExit(f"No eligible trials found for {name} in {sources}")
        selected[name] = trials

    if args.checkpoint_games_per_task <= 0:
        raise SystemExit("--checkpoint-games-per-task must be positive")
    rng = random.Random(args.sample_seed)
    checkpoint_trials: list[Trial] = []
    for task_type in ALL_TASK_TYPES:
        candidates = [
            trial for trial in selected["valid_all"] if trial.task_type == task_type
        ]
        if args.checkpoint_games_per_task > len(candidates):
            raise SystemExit(
                f"Cannot sample {args.checkpoint_games_per_task} {task_type} games "
                f"from a pool containing {len(candidates)}"
            )
        checkpoint_trials.extend(
            rng.sample(candidates, args.checkpoint_games_per_task)
        )
    rng.shuffle(checkpoint_trials)
    checkpoint_size = args.checkpoint_games_per_task * len(ALL_TASK_TYPES)
    checkpoint_split = f"valid_task_balanced{checkpoint_size}"
    selected[checkpoint_split] = checkpoint_trials
    id_checkpoint_trials = [
        trial for trial in checkpoint_trials if trial.task_type in ID_TASK_TYPES
    ]
    id_checkpoint_size = args.checkpoint_games_per_task * len(ID_TASK_TYPES)
    id_checkpoint_split = f"valid_id_task_balanced{id_checkpoint_size}"
    selected[id_checkpoint_split] = id_checkpoint_trials

    for name, trials in selected.items():
        print(f"{name}: {len(trials)} trials {task_counts(trials)}")

    if args.dry_run:
        return

    if output_root.exists() and not args.replace:
        raise SystemExit(
            f"Refusing to overwrite existing output root: {output_root}\n"
            "Use --replace or choose a different --output-root."
        )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    backup_root = output_root.with_name(f".{output_root.name}.backup-{os.getpid()}")
    try:
        manifest: dict[str, object] = {
            "source_root": str(source_root),
            "validation_sources": [str(path) for path in validation_roots],
            "id_task_types": list(ID_TASK_TYPES),
            "ood_task_types": list(OOD_TASK_TYPES),
            "checkpoint_evaluation": {
                "split": checkpoint_split,
                "id_split": id_checkpoint_split,
                "seed": args.sample_seed,
                "total_count": checkpoint_size,
                "id_count": id_checkpoint_size,
                "games_per_task": args.checkpoint_games_per_task,
                "task_counts": task_counts(checkpoint_trials),
                "games": [
                    {
                        "path": trial.manifest_path,
                        "task_type": trial.task_type,
                    }
                    for trial in checkpoint_trials
                ],
            },
            "splits": {},
        }
        for name, trials in selected.items():
            destination_split = temporary_root / name
            destination_split.mkdir()
            sources = sorted({trial.source_split for trial in trials})
            include_source = len(sources) > 1
            for trial in trials:
                link_trial(trial, destination_split, include_source)
            manifest["splits"][name] = {
                "sources": [str(source) for source in sources],
                "trial_count": len(trials),
                "task_counts": task_counts(trials),
            }

        with (temporary_root / "split_manifest.json").open("w") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")

        if output_root.exists():
            if backup_root.exists():
                raise RuntimeError(f"Temporary backup path already exists: {backup_root}")
            output_root.rename(backup_root)
        try:
            temporary_root.rename(output_root)
        except BaseException:
            if backup_root.exists() and not output_root.exists():
                backup_root.rename(output_root)
            raise
        if backup_root.exists():
            shutil.rmtree(backup_root)
    except BaseException:
        if temporary_root.exists():
            shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    print(f"Created LaMer ALFWorld splits at {output_root}")


if __name__ == "__main__":
    main()
