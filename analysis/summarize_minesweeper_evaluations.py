#!/usr/bin/env python3
"""Build JSON, CSV, and Markdown summaries from checkpoint evaluation logs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
METRIC = re.compile(
    r"'(?P<key>val/[^']+)':\s*(?:np\.float64\()?"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def parse_metrics(log_path: Path) -> dict[str, float]:
    text = ANSI_ESCAPE.sub("", log_path.read_text(encoding="utf-8", errors="replace"))
    metrics: dict[str, float] = {}
    for match in METRIC.finditer(text):
        metrics[match.group("key")] = float(match.group("value"))
    return metrics


def read_elapsed(path: Path) -> float | None:
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None


def checkpoint_step(directory: Path) -> int:
    if directory.name == "baseline":
        return 0
    match = re.fullmatch(r"step_(\d+)", directory.name)
    if not match:
        raise ValueError(f"Unexpected evaluation directory name: {directory.name}")
    return int(match.group(1))


def load_diagnostic_summary(directory: Path) -> tuple[dict, str | None]:
    candidates = sorted((directory / "validation_diagnostics").glob("step_*_summary.json"))
    if not candidates:
        return {}, None
    path = candidates[-1]
    return json.loads(path.read_text(encoding="utf-8")), str(path)


def diagnostic_value(metrics: dict, name: str):
    return metrics.get(f"val/diagnostics/{name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    candidates = [path for path in args.eval_dir.iterdir() if path.is_dir()]
    for directory in sorted(
        candidates,
        key=lambda path: checkpoint_step(path)
        if path.name == "baseline" or path.name.startswith("step_")
        else 10**12,
    ):
        if directory.name != "baseline" and not re.fullmatch(r"step_\d+", directory.name):
            continue
        log_path = directory / "eval.log"
        if not log_path.exists():
            continue
        metrics = parse_metrics(log_path)
        diagnostic_summary, diagnostic_summary_path = load_diagnostic_summary(directory)
        diagnostic_metrics = diagnostic_summary.get("metrics", {})
        # Prefer the lossless JSON values; fall back to scalars parsed from logs.
        combined_metrics = {**metrics, **diagnostic_metrics}
        row = {
            "checkpoint": directory.name,
            "global_step": checkpoint_step(directory),
            "pass_at_1": metrics.get("val/success_rate[0]"),
            "pass_at_2": metrics.get("val/success_rate[1]"),
            "pass_at_3": metrics.get("val/success_rate[2]"),
            "mean_reward": metrics.get("val/test_score/text"),
            "play_records": diagnostic_value(combined_metrics, "play_record_count"),
            "all_record_parse_rate": diagnostic_value(combined_metrics, "all_record_parse_rate"),
            "play_action_tag_rate": diagnostic_value(combined_metrics, "play_complete_action_tag_rate"),
            "play_parse_rate": diagnostic_value(combined_metrics, "play_parse_rate"),
            "play_effective_rate": diagnostic_value(combined_metrics, "play_effective_rate"),
            "parsed_but_ineffective_rate": diagnostic_value(combined_metrics, "parsed_but_ineffective_rate"),
            "play_token_limit_rate": diagnostic_value(combined_metrics, "play_response_token_limit_rate"),
            "reflection_parse_rate": diagnostic_value(combined_metrics, "reflection_parse_rate"),
            "tasks_with_zero_parsed_action_rate": diagnostic_value(
                combined_metrics, "tasks_with_zero_parsed_action_rate"
            ),
            "win_transitions": diagnostic_value(combined_metrics, "win_transition_count"),
            "terminal_losses": diagnostic_value(combined_metrics, "terminal_loss_count"),
            "attempt_1_parse_rate": diagnostic_value(combined_metrics, "attempt_1_parse_rate"),
            "attempt_2_parse_rate": diagnostic_value(combined_metrics, "attempt_2_parse_rate"),
            "attempt_3_parse_rate": diagnostic_value(combined_metrics, "attempt_3_parse_rate"),
            "attempt_1_effective_rate": diagnostic_value(combined_metrics, "attempt_1_effective_rate"),
            "attempt_2_effective_rate": diagnostic_value(combined_metrics, "attempt_2_effective_rate"),
            "attempt_3_effective_rate": diagnostic_value(combined_metrics, "attempt_3_effective_rate"),
            "wall_time_seconds": read_elapsed(directory / "elapsed_seconds.txt"),
            "status": "complete"
            if all(f"val/success_rate[{index}]" in metrics for index in range(3))
            else "failed",
            "diagnostic_summary_path": diagnostic_summary_path,
        }
        rows.append(row)
        payload = {**row, "validation_diagnostics": diagnostic_summary or None}
        (directory / "metrics.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

    fields = list(rows[0]) if rows else ["checkpoint", "global_step", "status"]
    with (args.eval_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    (args.eval_dir / "summary.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )

    def number(row: dict, key: str, digits: int = 4) -> str:
        value = row.get(key)
        return "" if value is None else f"{value:.{digits}f}"

    markdown = [
        "| Checkpoint | Step | pass@1 | pass@2 | pass@3 | Mean reward | Time (min) | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        elapsed = row["wall_time_seconds"]
        markdown.append(
            f"| {row['checkpoint']} | {row['global_step']} | {number(row, 'pass_at_1')} | "
            f"{number(row, 'pass_at_2')} | {number(row, 'pass_at_3')} | "
            f"{number(row, 'mean_reward')} | "
            f"{'' if elapsed is None else f'{elapsed / 60:.1f}'} | {row['status']} |"
        )

    markdown.extend([
        "",
        "## Validation interaction diagnostics",
        "",
        "| Checkpoint | Play records | All-record parse | Play action tag | Play parse | Play effective | Parsed but ineffective | Hit 1024 tokens | Reflection parse | Tasks with zero parsed plays |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        markdown.append(
            f"| {row['checkpoint']} | {'' if row['play_records'] is None else int(row['play_records'])} | "
            f"{number(row, 'all_record_parse_rate')} | {number(row, 'play_action_tag_rate')} | "
            f"{number(row, 'play_parse_rate')} | {number(row, 'play_effective_rate')} | "
            f"{number(row, 'parsed_but_ineffective_rate')} | {number(row, 'play_token_limit_rate')} | "
            f"{number(row, 'reflection_parse_rate')} | "
            f"{number(row, 'tasks_with_zero_parsed_action_rate')} |"
        )

    markdown.extend([
        "",
        "`All-record parse` matches the record-weighted training `valid_action_ratio` semantics and includes reflections. `Play parse` means the response passed the Minesweeper `<action>(row, col)</action>` parser. `Play effective` means the visible board changed. Detailed prompts, responses, boards, reflections, parsed actions, rewards, and terminal outcomes are in each checkpoint's `validation_diagnostics/*_interactions.jsonl`.",
    ])
    (args.eval_dir / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} evaluation rows to {args.eval_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
