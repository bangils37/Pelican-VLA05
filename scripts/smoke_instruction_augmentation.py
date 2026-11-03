"""Read real frames and tokenize augmented instructions without loading policy weights."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'pelican_vla0.5_infer/src')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from training.dataset import LeRobotPelicanDataset, collate_pelican_batch
from training.instruction_augmentation import CANONICAL_HIGH_LEVEL_TASK, CANONICAL_SUBTASKS


def main():
    if not __debug__:
        raise RuntimeError("Run this smoke test without -O so assertions stay enabled")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_path', default='/home/aitt/data/serving_brewed_coffee_clean_374_episodes')
    parser.add_argument('--qwen3_vl_path', default='Qwen/Qwen3-VL-4B-Instruct')
    args = parser.parse_args()
    torch.set_num_threads(2)
    random.seed(42)
    torch.manual_seed(42)
    train = LeRobotPelicanDataset(args.dataset_path, qwen3_vl_path=args.qwen3_vl_path,
                                  instruction_augmentation=True, augmentation_prob=1)
    val = LeRobotPelicanDataset(args.dataset_path, qwen3_vl_path=args.qwen3_vl_path,
                                instruction_augmentation=True, split='val',
                                _shared_data=train.get_shared_data())
    assert train.active_episodes_set.isdisjoint(val.active_episodes_set)
    # Exercise the actual dataset selection path for every label and boundary policy.
    for sub in range(5):
        assert val.select_instruction(0, sub) == CANONICAL_HIGH_LEVEL_TASK
    val.eval_instruction_level = 'subtask'
    for sub in range(5):
        assert val.select_instruction(0, sub) == CANONICAL_SUBTASKS[sub]
    train.augmentation_prob = 0
    train.use_subtasks_prob = 1
    assert train.select_instruction(0, 2, np.array([2, 3, 2])) == CANONICAL_HIGH_LEVEL_TASK
    assert train.select_instruction(0, 2, np.array([2, 2])) == CANONICAL_SUBTASKS[2]
    # Exercise boundary handling through real __getitem__, verifying unchanged targets.
    boundary_index = next(i for i, (row, _, frame, length, _) in enumerate(train.samples)
                          if frame + 49 < length and train.subtask_indices[row] >= 0
                          and train.subtask_indices[row] != train.subtask_indices[row + 49])
    before = train[boundary_index]
    train.subtask_boundary_policy = 'high_level'
    after = train[boundary_index]
    assert before['task'] != CANONICAL_HIGH_LEVEL_TASK
    assert after['task'] == CANONICAL_HIGH_LEVEL_TASK
    assert torch.equal(before['action'], after['action'])
    train.subtask_boundary_policy = 'keep'
    train.augmentation_prob = 1
    all_text = list(train.augmenter.high_level_pool)
    for pool in train.augmenter.subtask_pools.values():
        all_text.extend(pool)
    tokenizer = train.processor.processor.tokenizer
    max_tokens = max(len(tokenizer(t, truncation=False).input_ids) for t in all_text)
    assert max_tokens <= train.processor.max_length, max_tokens
    indices = []
    for sub in range(5):
        indices.append(next(i for i, sample in enumerate(train.samples) if train.subtask_indices[sample[0]] == sub))
    # Direct __getitem__ tests actual decoded images and tokenization.
    for i in indices:
        sample = train[i]
        assert sample['task'] in train.augmenter.subtask_pools[int(train.subtask_indices[train.samples[i][0]])]
        for slot in range(3):
            assert sample[f'observation.images.image{slot}'].abs().sum() > 0
        assert sample['action'].shape == (50, 32)
    # Close parent video handles before starting worker processes.
    train._video_readers.clear()
    def batch_once():
        loader = DataLoader(Subset(train, indices), batch_size=2, num_workers=2,
                            collate_fn=collate_pelican_batch, generator=torch.Generator().manual_seed(42))
        return list(loader)
    first_epoch, second_epoch = batch_once(), batch_once()
    assert len(first_epoch) == len(second_epoch) == 3
    for first, second in zip(first_epoch, second_epoch):
        assert torch.equal(first['observation.task.input_ids'], second['observation.task.input_ids'])
        assert torch.isfinite(first['action']).all()
    first = first_epoch[0]
    assert first['action'].shape == (2, 50, 32)
    val_first, val_second = val[0], val[0]
    assert torch.equal(val_first['observation.task.input_ids'], val_second['observation.task.input_ids'])
    with tempfile.TemporaryDirectory() as d:
        train.save_instruction_config(d)
        config = json.loads((Path(d) / 'instruction_augmentation.json').read_text())
        assert config['pool_sha256'] == train.augmenter.pool_sha256
    print(json.dumps({'train_episodes':len(train.active_episodes_set), 'val_episodes':len(val.active_episodes_set),
                      'train_frames':len(train), 'val_frames':len(val), 'pool':train.augmenter.stats(),
                      'max_instruction_tokens':max_tokens, 'batch_action_shape':list(first['action'].shape),
                      'two_worker_reproducible':True, 'validation_repeatable':True}, indent=2))


if __name__ == '__main__':
    main()
