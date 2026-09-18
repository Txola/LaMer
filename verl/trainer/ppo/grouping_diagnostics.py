"""Read-only Minesweeper grouping diagnostics, written before actor backward."""

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from verl.trainer.ppo.core_gigpo import to_hashable


def _json_value(value):
    """Collation can turn coordinate lists and numeric anchors into arrays."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported diagnostic JSON value: {type(value)}")


def dump_gigpo_groups(batch, output_dir, step, cross_attempt_only=False):
    """Describe the optimization records without changing their training groups.

    Attempt indices in output are one-based; turn indices remain zero-based.
    Repeated (traj_uid, phase, traj_idx, turn_idx) values identify batch copies.
    Reflection anchors/records are excluded from all play-group statistics.
    """
    metadata = batch.non_tensor_batch
    required = ('uid', 'traj_uid', 'traj_idx', 'turn_idx', 'phase',
                'anchor_obs', 'previous_reflections')
    for field in required:
        if field not in metadata:
            raise ValueError(f"GiGPO diagnostics require rollout metadata: {field}")

    # Only scalar returns leave tensor storage. Reflection text stays on CPU.
    step_returns = batch.batch['step_returns'].detach().cpu().tolist()
    groups = defaultdict(list)
    excluded_reflection_records = 0
    excluded_other_records = 0
    for i, phase in enumerate(metadata['phase']):
        anchor = metadata['anchor_obs'][i]
        anchor_key = to_hashable(anchor)
        if phase == 'reflect' or anchor_key == 'reflection':
            excluded_reflection_records += 1
            continue
        if phase != 'play':
            excluded_other_records += 1
            continue
        groups[(metadata['uid'][i], anchor_key)].append(i)

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    prefix = f"step_{step:06d}"
    groups_path = output_path / f"{prefix}_groups.jsonl"
    records_path = output_path / f"{prefix}_play_records.jsonl"
    summary_path = output_path / f"{prefix}_summary.json"
    non_singleton_count = 0
    cross_attempt_count = 0
    cross_attempt_records = 0
    written_group_count = 0
    histogram = Counter()

    def optional_value(field, i):
        return metadata[field][i] if field in metadata else None

    def member_record(i):
        parsing_valid = optional_value('is_action_valid', i)
        effective = optional_value('action_is_effective', i)
        reward = optional_value('rewards', i)
        return {
            'traj_uid': str(metadata['traj_uid'][i]),
            'phase': 'play',
            'traj_idx': int(metadata['traj_idx'][i]) + 1,
            'turn_idx': int(metadata['turn_idx'][i]),
            'previous_reflections': list(metadata['previous_reflections'][i]),
            'step_return': float(step_returns[i]),
            'parsed_action': optional_value('parsed_action', i),
            'is_action_valid': bool(parsing_valid) if parsing_valid is not None else None,
            'action_is_effective': bool(effective) if effective is not None else None,
            'immediate_reward': float(reward) if reward is not None else None,
            'next_anchor_obs': optional_value('next_anchor_obs', i),
        }

    # Full play population for structural analysis, independent of the detail
    # filter and including singletons and adjust_batch copies.
    with records_path.open('w', encoding='utf-8') as stream:
        for (uid, _), indices in groups.items():
            for i in indices:
                record = {
                    'record_index': int(i),
                    'uid': str(uid),
                    'anchor_obs': metadata['anchor_obs'][i],
                    **member_record(i),
                }
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False, default=_json_value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())

    with groups_path.open('w', encoding='utf-8') as stream:
        for (uid, _), indices in groups.items():
            size = len(indices)
            histogram[size] += 1
            attempt_counts = Counter(int(metadata['traj_idx'][i]) + 1 for i in indices)
            cross_attempt = len(attempt_counts) > 1
            if size == 1:
                continue
            non_singleton_count += 1
            if cross_attempt:
                cross_attempt_count += 1
                cross_attempt_records += size
            if cross_attempt_only and not cross_attempt:
                continue
            record = {
                'uid': str(uid),
                'anchor_obs': metadata['anchor_obs'][indices[0]],
                'phase': 'play',
                'group_size': size,
                'distinct_trajectory_count': len({metadata['traj_uid'][i] for i in indices}),
                'attempt_counts': {str(k): v for k, v in sorted(attempt_counts.items())},
                'members': [member_record(i) for i in indices],
            }
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False, default=_json_value) + '\n')
            written_group_count += 1
        stream.flush()
        os.fsync(stream.fileno())

    play_record_count = sum(len(indices) for indices in groups.values())
    summary = {
        'schema_version': 2,
        'play_records_file': records_path.name,
        'play_records_include_all_play_groups': True,
        'member_metadata_semantics': {
            'parsed_action': 'Exact parser coordinates sent to the environment; left click is implicit.',
            'is_action_valid': 'Parsing validity only; not board bounds or action effectiveness.',
            'action_is_effective': 'Environment flag comparing board_disp to board_disp_prev; retained verbatim.',
            'immediate_reward': 'Raw environment reward before trainer penalties; not step_return.',
            'next_anchor_obs': 'Visible board returned after this action.',
            'missing_optional_fields': 'null; unavailable, not false or zero.',
        },
        'global_step': int(step),
        'record_view': 'optimization_batch_including_adjust_batch_copies',
        'attempt_index_base': 1,
        'turn_index_base': 0,
        'reflection_groups_excluded': True,
        'excluded_reflection_records': excluded_reflection_records,
        'excluded_other_records': excluded_other_records,
        'total_play_records': play_record_count,
        'total_play_groups': len(groups),
        'non_singleton_play_groups': non_singleton_count,
        'cross_attempt_non_singleton_play_groups': cross_attempt_count,
        'fraction_non_singleton_play_groups_spanning_multiple_attempts': (
            cross_attempt_count / non_singleton_count if non_singleton_count else 0.0
        ),
        'play_records_in_cross_attempt_groups': cross_attempt_records,
        'fraction_play_records_in_cross_attempt_groups': (
            cross_attempt_records / play_record_count if play_record_count else 0.0
        ),
        'group_size_histogram': {str(k): v for k, v in sorted(histogram.items())},
        'cross_attempt_only': bool(cross_attempt_only),
        'detailed_groups_written': written_group_count,
    }
    with summary_path.open('w', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    return output_path
