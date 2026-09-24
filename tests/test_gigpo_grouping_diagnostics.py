import copy
import json
import tempfile
import unittest
from functools import partial
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from agent_system.environments.minesweeper.env_manager import MineSweeperEnvironmentManager
from agent_system.environments.minesweeper.prompt import parse_reflection
from agent_system.environments.minesweeper.projection import minesweeper_projection
from agent_system.multi_turn_rollout.utils import adjust_batch, to_list_of_dict
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.trainer.ppo.grouping_diagnostics import dump_gigpo_groups


def object_array(values):
    array = np.empty(len(values), dtype=object)
    for i, value in enumerate(values):
        array[i] = value
    return array


class GroupingDiagnosticsTest(unittest.TestCase):
    def make_batch(self):
        # A contains a cross-attempt pair and a copied interaction. B contains
        # two same-attempt records. C is singleton; reflection is excluded.
        metadata = {
            'uid': ['task'] * 7,
            'traj_uid': ['a', 'b', 'b', 'a', 'b', 'c', 'a'],
            'traj_idx': [0, 1, 1, 0, 0, 2, 1],
            'turn_idx': [0, 0, 0, 1, 1, 2, 0],
            'phase': ['play'] * 6 + ['reflect'],
            'anchor_obs': ['Row 1: ? .'] * 3 + ['Row 1: 1 .'] * 2 + ['C', 'reflection'],
            'previous_reflections': [[], ['Avoid (1, 1).\nTry (2, 2).'],
                                     ['Avoid (1, 1).\nTry (2, 2).'], [], [], ['r0', 'r1'], []],
            'parsed_action': [(-1, -1), ['1', '2'], ['1', '2'], ['2', '2'],
                              ['3', '3'], ['4', '4'], None],
            'is_action_valid': [False, True, True, True, True, True, True],
            'action_is_effective': [False, False, False, True, True, True, None],
            'rewards': [-1., -1., -1., .5, 2., 10., 1.],
            'next_anchor_obs': ['Row 1: ? .'] * 3 + ['changed'] * 3 + [None],
        }
        return DataProto.from_dict(
            tensors={
                'input_ids': torch.zeros((7, 2), dtype=torch.long),
                'step_returns': torch.tensor([1.7, 2., 2., 0., 1., -1., 0.]),
                'advantages': torch.arange(14, dtype=torch.float32).reshape(7, 2),
            },
            non_tensors={key: object_array(values) for key, values in metadata.items()},
        )

    def test_snapshot_matches_prompt_modes_and_is_taken_by_value(self):
        manager = MineSweeperEnvironmentManager.__new__(MineSweeperEnvironmentManager)
        manager.num_processes = 1
        manager.reflection_type = 'reflection_only'
        manager.reflections = [{0: 'r0', 1: 'r1'}]
        for attempt, expected in [(0, []), (1, ['r0']), (2, ['r0', 'r1'])]:
            manager.curr_traj_idx = attempt
            snapshot = manager.get_previous_reflections('play')[0]
            self.assertEqual(snapshot, expected)
            rendered = parse_reflection(attempt, {0: 'h0', 1: 'h1'},
                                        manager.reflections[0], manager.reflection_type)
            for reflection in snapshot:
                self.assertIn(reflection, rendered)
        snapshot = manager.get_previous_reflections()[0]
        manager.reflections[0][0] = 'changed'
        self.assertEqual(snapshot, ['r0', 'r1'])
        self.assertEqual(manager.get_previous_reflections('reflect'), [[]])
        manager.reflection_type = 'history_only'
        self.assertEqual(manager.get_previous_reflections(), [[]])
        manager.reflection_type = 'history_and_reflection'
        self.assertEqual(manager.get_previous_reflections(), [['changed', 'r1']])
        manager.reflections = [{}]
        self.assertEqual(manager.get_previous_reflections(), [[]])

    def test_groups_summary_and_no_training_mutation(self):
        batch = self.make_batch()
        before_tensors = {key: value.clone() for key, value in batch.batch.items()}
        before_metadata = copy.deepcopy(batch.non_tensor_batch)
        torch_rng = torch.random.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            dump_gigpo_groups(batch, directory, step=1)
            groups = [json.loads(line) for line in
                      (Path(directory) / 'step_000001_groups.jsonl').read_text().splitlines()]
            summary = json.loads((Path(directory) / 'step_000001_summary.json').read_text())
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]['anchor_obs'], 'Row 1: ? .')
        self.assertEqual(groups[0]['group_size'], 3)
        self.assertEqual(groups[0]['distinct_trajectory_count'], 2)
        self.assertEqual(groups[0]['attempt_counts'], {'1': 1, '2': 2})
        self.assertEqual(groups[0]['members'][0]['traj_idx'], 1)
        self.assertEqual(groups[0]['members'][1]['previous_reflections'],
                         ['Avoid (1, 1).\nTry (2, 2).'])
        self.assertEqual(groups[0]['members'][1], groups[0]['members'][2])
        self.assertEqual(groups[0]['members'][0]['parsed_action'], [-1, -1])
        self.assertFalse(groups[0]['members'][0]['is_action_valid'])
        self.assertEqual(groups[0]['members'][1]['parsed_action'], ['1', '2'])
        self.assertFalse(groups[0]['members'][1]['action_is_effective'])
        self.assertEqual(groups[0]['members'][1]['immediate_reward'], -1.)
        self.assertEqual(groups[0]['members'][1]['next_anchor_obs'], 'Row 1: ? .')
        self.assertEqual(summary['schema_version'], 2)
        self.assertEqual(summary['total_play_groups'], 3)
        self.assertEqual(summary['non_singleton_play_groups'], 2)
        self.assertEqual(summary['cross_attempt_non_singleton_play_groups'], 1)
        self.assertEqual(summary['fraction_non_singleton_play_groups_spanning_multiple_attempts'], .5)
        self.assertEqual(summary['fraction_play_records_in_cross_attempt_groups'], .5)
        self.assertEqual(summary['group_size_histogram'], {'1': 1, '2': 1, '3': 1})
        self.assertEqual(summary['excluded_reflection_records'], 1)
        for key, value in before_tensors.items():
            torch.testing.assert_close(batch.batch[key], value, rtol=0, atol=0)
        for key, value in before_metadata.items():
            self.assertEqual(batch.non_tensor_batch[key].tolist(), value.tolist())
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_rng))

    def test_copy_and_reorder_preserve_reflection_identity(self):
        batch = self.make_batch()
        batch = DataProto.from_single_dict(collate_fn(to_list_of_dict(batch)))
        config = OmegaConf.create({
            'actor_rollout_ref': {
                'rollout': {'log_prob_micro_batch_size_per_gpu': 1},
                'actor': {'ppo_mini_batch_size': 4},
            },
            'trainer': {'n_gpus_per_node': 1},
        })
        adjusted = adjust_batch(config, batch)
        adjusted.reorder(torch.arange(len(adjusted) - 1, -1, -1))
        with tempfile.TemporaryDirectory() as directory:
            dump_gigpo_groups(adjusted, directory, step=2, cross_attempt_only=True)
            groups = [json.loads(line) for line in
                      (Path(directory) / 'step_000002_groups.jsonl').read_text().splitlines()]
            summary = json.loads((Path(directory) / 'step_000002_summary.json').read_text())
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]['group_size'], 4)
        self.assertEqual(groups[0]['attempt_counts'], {'1': 2, '2': 2})
        self.assertEqual(summary['total_play_groups'], 3)
        for member in groups[0]['members']:
            self.assertEqual(member['previous_reflections'], [] if member['traj_idx'] == 1
                             else ['Avoid (1, 1).\nTry (2, 2).'])

    def test_complete_records_include_singletons_and_filtered_groups(self):
        batch = self.make_batch()
        with tempfile.TemporaryDirectory() as directory:
            dump_gigpo_groups(batch, directory, step=1, cross_attempt_only=True)
            records = [json.loads(line) for line in
                       (Path(directory) / 'step_000001_play_records.jsonl').read_text().splitlines()]
            summary = json.loads((Path(directory) / 'step_000001_summary.json').read_text())
        self.assertEqual(len(records), 6)
        self.assertEqual({record['record_index'] for record in records}, set(range(6)))
        self.assertEqual(summary['play_records_file'], 'step_000001_play_records.jsonl')
        self.assertTrue(summary['play_records_include_all_play_groups'])
        self.assertEqual(summary['detailed_groups_written'], 1)
        self.assertEqual(records[-1]['anchor_obs'], 'C')
        self.assertEqual(records[-1]['immediate_reward'], 10.)

    def test_optional_metadata_is_unavailable_not_false_or_zero(self):
        batch = self.make_batch()
        for field in ('parsed_action', 'is_action_valid', 'action_is_effective', 'rewards', 'next_anchor_obs'):
            del batch.non_tensor_batch[field]
        with tempfile.TemporaryDirectory() as directory:
            dump_gigpo_groups(batch, directory, step=1)
            group = json.loads((Path(directory) / 'step_000001_groups.jsonl').read_text().splitlines()[0])
        for field in ('parsed_action', 'is_action_valid', 'action_is_effective', 'immediate_reward', 'next_anchor_obs'):
            self.assertIsNone(group['members'][0][field])

    def test_uniform_coordinate_lists_survive_real_collation(self):
        batch = self.make_batch().select_idxs([1, 2])
        batch = DataProto.from_single_dict(collate_fn(to_list_of_dict(batch)))
        self.assertEqual(batch.non_tensor_batch['parsed_action'].shape, (2, 2))
        with tempfile.TemporaryDirectory() as directory:
            dump_gigpo_groups(batch, directory, step=1)
            group = json.loads((Path(directory) / 'step_000001_groups.jsonl').read_text().strip())
        self.assertEqual(group['members'][0]['parsed_action'], ['1', '2'])

    def test_manager_captures_parser_result_without_changing_step_behavior(self):
        class FakeEnvs:
            num_processes = 1

            def reset(self):
                return ['Row 1: ? ?'], [{'won': False}]

            def step(self, actions):
                self.received_actions = copy.deepcopy(actions)
                return ['Row 1: ? ?'], [-1.], [False], [{'won': False, 'action_is_effective': True}]

        for response in ('unparseable', '<action>(9, 9)</action>'):
            outputs = []
            for enabled in (False, True):
                config = OmegaConf.create({
                    'env': {'minesweeper': {'n_mines': 1, 'board_size': 2}},
                    'trainer': {'grouping_diagnostics': {'enabled': enabled}},
                    'algorithm': {'adv_estimator': 'gigpo'},
                })
                envs = FakeEnvs()
                manager = MineSweeperEnvironmentManager(
                    envs, partial(minesweeper_projection, board_size=2), 3, True, config)
                manager.reset()
                obs, rewards, dones, infos = manager.step([response])
                parsed = infos[0].pop('diagnostic_parsed_action', None)
                if enabled:
                    self.assertEqual(parsed, list(envs.received_actions[0]))
                    self.assertEqual(bool(infos[0]['is_action_valid']), response != 'unparseable')
                else:
                    self.assertIsNone(parsed)
                outputs.append((obs, rewards.tolist(), dones.tolist(), infos, envs.received_actions))
            self.assertEqual(outputs[0], outputs[1])

    def test_existing_anchor_equality_and_task_boundaries(self):
        batch = self.make_batch().select_idxs([0, 1, 2])
        batch.non_tensor_batch['anchor_obs'] = object_array([[1, 2], np.array([1, 2]), [1, 2]])
        batch.non_tensor_batch['uid'][2] = 'other-task'
        with tempfile.TemporaryDirectory() as directory:
            dump_gigpo_groups(batch, directory, step=5)
            groups = [json.loads(line) for line in
                      (Path(directory) / 'step_000005_groups.jsonl').read_text().splitlines()]
            summary = json.loads((Path(directory) / 'step_000005_summary.json').read_text())
        self.assertEqual(summary['total_play_groups'], 2)
        self.assertEqual(groups[0]['group_size'], 2)

    def test_no_play_records_and_missing_metadata(self):
        batch = self.make_batch().select_idxs([6])
        with tempfile.TemporaryDirectory() as directory:
            dump_gigpo_groups(batch, directory, step=3)
            summary = json.loads((Path(directory) / 'step_000003_summary.json').read_text())
            self.assertEqual(summary['total_play_groups'], 0)
            self.assertEqual(summary['fraction_play_records_in_cross_attempt_groups'], 0.)
            self.assertEqual((Path(directory) / 'step_000003_groups.jsonl').read_text(), '')
            del batch.non_tensor_batch['previous_reflections']
            with self.assertRaisesRegex(ValueError, 'previous_reflections'):
                dump_gigpo_groups(batch, directory, step=4)


if __name__ == '__main__':
    unittest.main()
