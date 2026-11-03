"""Regression tests for instruction selection and reproducibility (no model download)."""
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.instruction_augmentation import (
    DynamicCoffeeInstructionAugmenter, CANONICAL_HIGH_LEVEL_TASK, CANONICAL_SUBTASKS,
)


class InstructionTests(unittest.TestCase):
    def setUp(self):
        self.aug = DynamicCoffeeInstructionAugmenter()

    def test_eval_never_consumes_randomness(self):
        state = random.getstate()
        for split in ('val', 'test', 'all'):
            for sub in range(-1, 5):
                self.assertEqual(self.aug.sample_instruction(subtask_idx=sub, split=split), CANONICAL_HIGH_LEVEL_TASK)
                self.assertEqual(self.aug.sample_instruction(subtask_idx=sub, split=split, eval_instruction_level='subtask'), CANONICAL_SUBTASKS.get(sub, CANONICAL_HIGH_LEVEL_TASK))
        self.aug.sample_instruction(subtask_idx=2, deterministic=True)
        self.assertEqual(random.getstate(), state)

    def test_canonical_and_paraphrase_are_separate_controls(self):
        for sub in range(5):
            for _ in range(30):
                self.assertEqual(self.aug.sample_instruction(subtask_idx=sub, use_subtasks_prob=1, augmentation_prob=0), CANONICAL_SUBTASKS[sub])
                text = self.aug.sample_instruction(subtask_idx=sub, use_subtasks_prob=1, augmentation_prob=1)
                self.assertIn(text, self.aug.subtask_pools[sub])
                self.assertNotEqual(text, CANONICAL_SUBTASKS[sub])
        self.assertEqual(self.aug.sample_instruction(subtask_idx=-1, use_subtasks_prob=1, augmentation_prob=0), CANONICAL_HIGH_LEVEL_TASK)

    def test_seed_reproduces_sequence_and_distribution(self):
        def sequence():
            rng = random.Random(42)
            return [self.aug.sample_instruction(subtask_idx=2, use_subtasks_prob=1, rng=rng) for _ in range(2000)]
        a, b = sequence(), sequence()
        self.assertEqual(a, b)
        canonical_fraction = a.count(CANONICAL_SUBTASKS[2]) / len(a)
        self.assertTrue(.45 < canonical_fraction < .55)
        self.assertGreater(len(set(a)), 3)

    def test_load_custom_pool_and_roundtrip_hash(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'pool.json'
            data = self.aug.to_dict()
            data['high_level'] = ['Please carry out the coffee serving task.']
            path.write_text(json.dumps(data))
            loaded = DynamicCoffeeInstructionAugmenter(path)
            self.assertEqual(loaded.sample_instruction(augmentation_prob=1), data['high_level'][0])
            loaded.save_pool(path)
            self.assertEqual(loaded.pool_sha256, DynamicCoffeeInstructionAugmenter(path).pool_sha256)
            with self.assertRaises(FileNotFoundError):
                DynamicCoffeeInstructionAugmenter(Path(d) / 'missing.json')

    def test_invalid_pool_fails_without_partial_update(self):
        original_hash = self.aug.pool_sha256
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'pool.json'
            for invalid in ([], '', [5], [''], ['line\nbreak']):
                data = self.aug.to_dict()
                data['subtasks']['2'] = invalid
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    self.aug.load_pool(path)
                self.assertEqual(self.aug.pool_sha256, original_hash)

    def test_reject_wrong_ids_and_invalid_probabilities(self):
        for kwargs in ({'task_idx':1}, {'subtask_idx':5}, {'augmentation_prob':-1}, {'use_subtasks_prob':float('nan')}, {'split':'typo'}):
            with self.assertRaises(ValueError):
                self.aug.sample_instruction(**kwargs)

    def test_control_characters_and_punctuation_only_rejected(self):
        from training.instruction_augmentation import validate_pool
        for text in ('Serve\rcoffee.', 'Serve\tcoffee.', 'Serve\x00coffee.', 'Serve\u2028coffee.', '!!!'):
            data = self.aug.to_dict()
            data['high_level'] = [text]
            with self.subTest(text=repr(text)), self.assertRaises(ValueError):
                validate_pool(data)

    def test_canonical_only_pool_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'pool.json'
            path.write_text(json.dumps({'high_level':[CANONICAL_HIGH_LEVEL_TASK],
                'subtasks':{str(k):[v] for k,v in CANONICAL_SUBTASKS.items()}}))
            aug = DynamicCoffeeInstructionAugmenter(path)
            for sub in range(5):
                self.assertEqual(aug.sample_instruction(subtask_idx=sub, use_subtasks_prob=1,
                                 augmentation_prob=1), CANONICAL_SUBTASKS[sub])

    def test_joint_level_and_augmentation_distribution(self):
        rng = random.Random(77)
        samples = [self.aug.sample_instruction(subtask_idx=2, use_subtasks_prob=.3,
                   augmentation_prob=.6, rng=rng) for _ in range(10000)]
        sub = [x for x in samples if x in self.aug.subtask_pools[2]]
        high = [x for x in samples if x in self.aug.high_level_pool]
        self.assertEqual(len(sub)+len(high), 10000)
        self.assertAlmostEqual(len(sub)/10000, .3, delta=.025)
        self.assertAlmostEqual(sub.count(CANONICAL_SUBTASKS[2])/len(sub), .4, delta=.035)
        self.assertAlmostEqual(high.count(CANONICAL_HIGH_LEVEL_TASK)/len(high), .4, delta=.025)

    def test_pool_schema_rejects_missing_extra_or_wrong_keys(self):
        from training.instruction_augmentation import validate_pool
        for data in (None, [], {}, {'high_level':['Serve coffee.'], 'subtasks':{}},
                     dict(self.aug.to_dict(), unexpected=True)):
            with self.subTest(data=data), self.assertRaises(ValueError):
                validate_pool(data)


if __name__ == '__main__':
    unittest.main()
