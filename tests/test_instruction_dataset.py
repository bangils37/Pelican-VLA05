"""Synthetic Parquet integration tests. Uses the real loader, no network/GPU/model weights."""
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'pelican_vla0.5_infer/src')]
os.environ['HF_HUB_OFFLINE'] = '1'
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader
from training.dataset import LeRobotPelicanDataset
from training.instruction_augmentation import CANONICAL_HIGH_LEVEL_TASK as HIGH, CANONICAL_SUBTASKS as SUB, DynamicCoffeeInstructionAugmenter


class FakeProcessor:
    """Only replace image/token processing; actual indexing and selection stay intact."""
    def __init__(self, **kwargs):
        self.processor = self
        self.max_length = kwargs['max_length']

    def tokenizer(self, text, **kwargs):
        return SimpleNamespace(input_ids=list(range(len(text.split()))))

    def __call__(self, sample):
        return sample


def collect_text(batch):
    return [(item['task'], int(item['observation.state'][0])) for item in batch]


def make_fixture(root, ids=(0, 1, 2), version=3, labels=True):
    (root/'meta').mkdir(parents=True)
    (root/'data/chunk-000').mkdir(parents=True)
    (root/'meta/info.json').write_text(json.dumps({'fps':30}))
    pq.write_table(pa.table({'task':[HIGH], 'task_index':[0]}), root/'meta/tasks.parquet')
    pq.write_table(pa.table({'subtask':list(SUB.values()), 'subtask_index':list(SUB)}), root/'meta/subtasks.parquet')
    episodes, rows = [], []
    # 24 samples per episode; label changes and returns to catch endpoint-only checks.
    for n, eid in enumerate(ids):
        episodes.append({'episode_index':eid, 'length':24, 'dataset_from_index':24*n})
        data = {'observation.state':[[24*n+i, 0] for i in range(24)],
                'action':[[24*n+i, -i] for i in range(24)],
                'task_index':[0]*24}
        if labels:
            data['subtask_index'] = [0, 1, 0] + [2]*9 + [3]*6 + [4]*5 + [-1]
        table = pa.table(data)
        rows.append(table)
        if version == 2:
            pq.write_table(table, root/f'data/chunk-000/episode_{eid:06d}.parquet')
    if version == 3:
        (root/'meta/episodes/chunk-000').mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(episodes), root/'meta/episodes/chunk-000/file-000.parquet')
        pq.write_table(pa.concat_tables(rows), root/'data/chunk-000/file-000.parquet')
    else:
        (root/'meta/episodes.jsonl').write_text(''.join(json.dumps(ep)+'\n' for ep in episodes))


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'dataset'
        self.patch = patch('training.dataset.PelicanVLA05ProcessorTransformFn', FakeProcessor)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        torch.set_num_threads(1)

    def dataset(self, **kwargs):
        defaults = dict(camera_map={}, image_size=(8, 8), chunk_size=3, instruction_augmentation=True,
                        augmentation_prob=0, use_subtasks_prob=1, val_ratio=0)
        defaults.update(kwargs)
        return LeRobotPelicanDataset(self.root, **defaults)

    def test_sparse_episode_ids_and_split_partition(self):
        make_fixture(self.root, ids=(8, 2, 11))
        train = self.dataset(val_ratio=1/3)
        val = self.dataset(val_ratio=1/3, split='val', _shared_data=train.get_shared_data())
        self.assertEqual(len(train)+len(val), 72)
        self.assertEqual(train.active_episodes_set | val.active_episodes_set, {8, 2, 11})
        self.assertFalse(train.active_episodes_set & val.active_episodes_set)
        for ds in (train, val):
            for i, (row, *_rest) in enumerate(ds.samples):
                self.assertEqual(ds[i]['observation.state'][0], row)

    def test_boundary_inside_chunk_and_episode_end_padding(self):
        make_fixture(self.root)
        ds = self.dataset()
        self.assertEqual(ds[0]['task'], SUB[0])
        original = ds[0]['action'].clone()
        ds.subtask_boundary_policy = 'high_level'
        self.assertEqual(ds[0]['task'], HIGH)  # 0 -> 1 -> 0
        torch.testing.assert_close(ds[0]['action'], original)
        last = ds[23]
        self.assertEqual(last['task'], HIGH)
        self.assertEqual(last['action'][:,0].tolist(), [23, 23, 23])
        self.assertEqual(ds[24]['action'][:,0].tolist(), [24, 25, 26])

    def test_v2_empty_shared_cache_and_subtasks(self):
        make_fixture(self.root, version=2)
        train = self.dataset()
        self.assertEqual(train.get_shared_data(), {})
        val = self.dataset(split='val', _shared_data=train.get_shared_data(), eval_instruction_level='subtask')
        self.assertEqual(val[0]['task'], SUB[0])
        self.assertEqual(val[23]['task'], HIGH)
        self.assertEqual(val[24]['action'][:,0].tolist(), [24, 25, 26])

    def test_missing_subtask_column_falls_back(self):
        for version in (2, 3):
            self.root = Path(self.temp.name)/str(version)
            make_fixture(self.root, version=version, labels=False)
            ds = self.dataset()
            self.assertEqual(ds[0]['task'], HIGH)

    def test_baseline_eval_repeatability(self):
        make_fixture(self.root)
        ds = self.dataset(instruction_augmentation=False, split='val', eval_instruction_level='subtask')
        self.assertEqual([ds[0]['task'] for _ in range(20)], [SUB[0]]*20)
        ds.eval_instruction_level = 'high_level'
        self.assertEqual(ds[0]['task'], HIGH)

    def test_invalid_ratio_and_tiny_split_rejected(self):
        make_fixture(self.root, ids=(0,))
        for ratio in (-.1, 1, 2, float('nan'), .05):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                self.dataset(val_ratio=ratio)

    def test_token_limit_and_wrong_metadata_rejected(self):
        make_fixture(self.root)
        with self.assertRaises(ValueError):
            self.dataset(max_length=2)
        pq.write_table(pa.table({'task':['unrelated task'], 'task_index':[0]}), self.root/'meta/tasks.parquet')
        with self.assertRaises(ValueError):
            self.dataset()

    def test_checkpoint_regular_best_and_retention(self):
        from training.train import save_checkpoint
        make_fixture(self.root)
        ds = self.dataset()
        class FakePolicy:
            def save_pretrained(self, directory):
                (Path(directory)/'config.json').write_text('{}')
        out = Path(self.temp.name)/'checkpoints'
        for step in (1, 2):
            save_checkpoint(FakePolicy(), out, step, ds)
        save_checkpoint(FakePolicy(), out, 2, ds, is_best=True, val_loss=.123)
        self.assertFalse((out/'checkpoint-000001').exists())
        for folder in (out/'checkpoint-000002', out/'best_model'):
            data = json.loads((folder/'instruction_augmentation.json').read_text())
            restored = DynamicCoffeeInstructionAugmenter(folder/'coffee_instruction_pool.json')
            self.assertEqual(data['pool_sha256'], restored.pool_sha256)
            self.assertEqual(data['active_episode_ids'], [0,1,2])
        self.assertEqual((out/'best_model/best_val_loss.txt').read_text(), '0.123000\n')

    def test_disabling_augmentation_removes_stale_snapshot(self):
        make_fixture(self.root)
        ds = self.dataset()
        out = Path(self.temp.name)/'snapshot'
        ds.save_instruction_config(out)
        ds.augmenter = None
        ds.save_instruction_config(out)
        self.assertFalse((out/'coffee_instruction_pool.json').exists())

    def test_unknown_data_labels_rejected(self):
        make_fixture(self.root)
        path = self.root/'data/chunk-000/file-000.parquet'
        original = pq.read_table(path)
        for column, value in (('task_index', 7), ('subtask_index', 9)):
            table = original.set_column(original.column_names.index(column), column,
                                        pa.array([value]*len(original)))
            pq.write_table(table, path)
            with self.assertRaises(ValueError):
                self.dataset()

    def test_augmentation_preserves_all_action_and_state_targets(self):
        make_fixture(self.root)
        baseline = self.dataset(instruction_augmentation=False)
        augmented = self.dataset(augmentation_prob=1, subtask_boundary_policy='high_level')
        for i in range(len(baseline)):
            before, after = baseline[i], augmented[i]
            torch.testing.assert_close(before['action'], after['action'])
            torch.testing.assert_close(before['observation.state'], after['observation.state'])

    def test_multiworker_spawn_and_fork_reproducibility_and_epoch_variation(self):
        make_fixture(self.root)
        ds = self.dataset(augmentation_prob=1)
        for context in ('spawn', 'fork'):
            if context not in multiprocessing.get_all_start_methods():
                continue
            with self.subTest(context=context):
                def epochs():
                    loader = DataLoader(ds, batch_size=12, num_workers=2, collate_fn=collect_text,
                                        multiprocessing_context=context, timeout=45,
                                        generator=torch.Generator().manual_seed(123))
                    return [[item for batch in loader for item in batch] for _ in range(2)]
                first, second = epochs(), epochs()
                self.assertEqual(first, second)
                self.assertNotEqual(first[0], first[1])
                self.assertEqual([row for _,row in first[0]], list(range(72)))


if __name__ == '__main__':
    unittest.main()
