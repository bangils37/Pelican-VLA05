"""Offline curated pool, online O(1) sampling; no network calls during training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

CANONICAL_HIGH_LEVEL_TASK = 'serving brewed coffee'
CANONICAL_SUBTASKS = {0: 'Pick a cup and place it on the wooden serving tray', 1: '(OPTIONAL) Re-align the coffee pourer for easier grasp', 2: 'Pick the coffee pourer and pour a shot to the cup', 3: 'Place the coffee pourer back the original place', 4: 'Pick the cup and place to the other side of the wooden serving tray'}

DEFAULT_POOL_FILE = Path(__file__).with_name('coffee_instruction_pool.json')


def probability(value: float, name: str) -> float:
    if not 0 <= value <= 1:
        raise ValueError(f'{name} must be in [0, 1], got {value}')
    return value


def validate_instruction(item: str) -> str:
    """Reject invisible controls, line separators and strings without language content."""
    if (not isinstance(item, str) or not item.strip() or len(item) > 240
            or any(unicodedata.category(c).startswith('C') or c in '\u2028\u2029' for c in item)
            or not any(c.isalpha() for c in item)):
        raise ValueError('Instructions must contain letters, be single-line, <= 240 characters, and have no control characters')
    return item.strip()


def validate_pool(data: dict) -> dict:
    """Validate structure, not physical equivalence; new wording needs human review."""
    if not isinstance(data, dict) or set(data) != {'high_level', 'subtasks'}:
        raise ValueError('Pool requires high_level and subtasks only')
    if not isinstance(data['subtasks'], dict) or set(data['subtasks']) != set(map(str, range(5))):
        raise ValueError('Pool must contain exactly subtask IDs 0..4')
    def clean(items, canonical):
        if not isinstance(items, list) or not items:
            raise ValueError('Each pool must be a nonempty list')
        result, seen = [], set()
        for item in [canonical, *items]:
            item = validate_instruction(item)
            key = re.sub(r'[^\w]+', ' ', item.casefold()).strip()
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result
    return {'high_level': clean(data['high_level'], CANONICAL_HIGH_LEVEL_TASK),
            'subtasks': {str(k): clean(data['subtasks'][str(k)], v) for k, v in CANONICAL_SUBTASKS.items()}}


class DynamicCoffeeInstructionAugmenter:
    def __init__(self, pool_file: str | Path | None = None):
        self.load_pool(Path(pool_file) if pool_file is not None else DEFAULT_POOL_FILE)

    def load_pool(self, json_path: str | Path):
        data = validate_pool(json.loads(Path(json_path).read_text(encoding='utf-8')))
        # Commit only after full validation; tuples prevent accidental mutation.
        self.high_level_pool = tuple(data['high_level'])
        self.subtask_pools = {int(k): tuple(v) for k, v in data['subtasks'].items()}
        self._high_variants = tuple(x for x in self.high_level_pool if x != CANONICAL_HIGH_LEVEL_TASK)
        self._sub_variants = {k: tuple(x for x in v if x != CANONICAL_SUBTASKS[k]) for k, v in self.subtask_pools.items()}

    def to_dict(self):
        return {'high_level': list(self.high_level_pool),
                'subtasks': {str(k): list(v) for k, v in self.subtask_pools.items()}}

    @property
    def pool_sha256(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def save_pool(self, json_path: str | Path):
        Path(json_path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

    def sample_instruction(self, task_idx=0, subtask_idx=-1, use_subtasks_prob=0.5,
                           split='train', deterministic=False, augmentation_prob=0.5,
                           eval_instruction_level='high_level', rng=None):
        probability(use_subtasks_prob, 'use_subtasks_prob')
        probability(augmentation_prob, 'augmentation_prob')
        if task_idx != 0:
            raise ValueError(f'Coffee pool supports task_index=0 only, got {task_idx}')
        if subtask_idx not in (-1, *CANONICAL_SUBTASKS):
            raise ValueError(f'Unknown subtask_index: {subtask_idx}')
        if split not in ('train', 'val', 'test', 'all'):
            raise ValueError(f'Unknown split: {split}')
        if eval_instruction_level not in ('high_level', 'subtask'):
            raise ValueError('eval_instruction_level must be high_level or subtask')
        if deterministic or split != 'train':
            return (CANONICAL_SUBTASKS.get(subtask_idx, CANONICAL_HIGH_LEVEL_TASK)
                    if eval_instruction_level == 'subtask' else CANONICAL_HIGH_LEVEL_TASK)
        rng = rng or random  # PyTorch seeds Python random separately in every worker.
        use_sub = subtask_idx >= 0 and rng.random() < use_subtasks_prob
        canonical = CANONICAL_SUBTASKS[subtask_idx] if use_sub else CANONICAL_HIGH_LEVEL_TASK
        variants = self._sub_variants[subtask_idx] if use_sub else self._high_variants
        if not variants or rng.random() >= augmentation_prob:
            return canonical
        return rng.choice(variants)

    def stats(self):
        return {'high_level_variations': len(self.high_level_pool),
                'total_subtask_variations': sum(map(len, self.subtask_pools.values())),
                **{f'subtask_{k}': len(v) for k, v in self.subtask_pools.items()}}
