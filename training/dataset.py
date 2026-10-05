"""Dataset loader for Pelican-VLA 0.5 compatible with LeRobot v2.1 and v3 datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from decord import VideoReader, cpu

from lerobot.policies.pelicanvla05 import PelicanVLA05ProcessorTransformFn
from lerobot.utils.constants import OBS_IMAGES


class LeRobotPelicanDataset(Dataset):
    """Dataset for training Pelican-VLA 0.5 on LeRobot v2.1/v3 formatted datasets.
    
    Extracts:
      - 3 temporal image frames per camera: [t-1, t, t_future] (T=3)
      - State at current frame t: [state_dim]
      - Action chunk from t to t + chunk_size: [chunk_size, action_dim]
      - Task instruction text
      - Normalized state and action using dataset stats
    """

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        camera_map: Mapping[str, str],  # e.g. {"cam_head": "image0", "cam_left_wrist": "image1", "cam_right_wrist": "image2"}
        chunk_size: int = 50,
        future_horizon: int = 25,  # frame index delta for future ground truth target
        qwen3_vl_path: str = "Qwen/Qwen3-VL-4B-Instruct",
        max_length: int = 48,
        image_size: tuple[int, int] = (224, 224),
        split: str = "train",
    ):
        super().__init__()
        self.root = Path(dataset_root).expanduser().resolve()
        self.camera_map = dict(camera_map)
        self.chunk_size = chunk_size
        self.future_horizon = future_horizon
        self.image_size = image_size

        # Load info.json
        info_path = self.root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"info.json not found at {info_path}")
        with open(info_path, "r", encoding="utf-8") as f:
            self.info = json.load(f)

        self.fps = self.info.get("fps", 30)

        # Load tasks
        self.tasks = {}
        tasks_path = self.root / "meta" / "tasks.jsonl"
        if tasks_path.is_file():
            with open(tasks_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line.strip())
                    self.tasks[item["task_index"]] = item["task"]

        # Load episodes metadata
        self.episodes = []
        episodes_path = self.root / "meta" / "episodes.jsonl"
        if episodes_path.is_file():
            with open(episodes_path, "r", encoding="utf-8") as f:
                for line in f:
                    self.episodes.append(json.loads(line.strip()))

        # Load stats
        self.state_mean = None
        self.state_std = None
        self.action_mean = None
        self.action_std = None
        self._load_stats()

        # Build index map: list of (episode_idx, frame_idx)
        self.samples = []
        for ep in self.episodes:
            ep_idx = ep["episode_index"]
            ep_len = ep["length"]
            for frame_idx in range(ep_len):
                self.samples.append((ep_idx, frame_idx, ep_len))

        # Initialize processor transform
        self.processor = PelicanVLA05ProcessorTransformFn(
            pretrained_model_name_or_path=qwen3_vl_path,
            max_length=max_length,
        )

        # Cache open video readers
        self._video_readers = {}
        self._parquet_cache = {}

    def _load_stats(self):
        stats_path = self.root / "meta" / "episodes_stats.jsonl"
        if stats_path.is_file():
            with open(stats_path, "r", encoding="utf-8") as f:
                # Use first line or aggregate
                data = json.loads(f.readline().strip())
                stats = data.get("stats", {})
                if "observation.state" in stats:
                    self.state_mean = np.array(stats["observation.state"]["mean"], dtype=np.float32)
                    self.state_std = np.array(stats["observation.state"]["std"], dtype=np.float32)
                    self.state_std[self.state_std < 1e-4] = 1.0
                if "action" in stats:
                    self.action_mean = np.array(stats["action"]["mean"], dtype=np.float32)
                    self.action_std = np.array(stats["action"]["std"], dtype=np.float32)
                    self.action_std[self.action_std < 1e-4] = 1.0

    def _get_parquet_data(self, ep_idx: int):
        if ep_idx not in self._parquet_cache:
            parquet_path = self.root / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
            table = pq.read_table(parquet_path)
            self._parquet_cache[ep_idx] = {
                "state": np.stack(table["observation.state"].to_numpy()),
                "action": np.stack(table["action"].to_numpy()),
                "task_index": table["task_index"].to_numpy(),
            }
        return self._parquet_cache[ep_idx]

    def _get_video_reader(self, cam_key: str, ep_idx: int):
        key = (cam_key, ep_idx)
        if key not in self._video_readers:
            video_path = self.root / "videos" / "chunk-000" / f"observation.images.{cam_key}" / f"episode_{ep_idx:06d}.mp4"
            if not video_path.is_file():
                # try alternative path without prefix
                video_path = self.root / "videos" / "chunk-000" / cam_key / f"episode_{ep_idx:06d}.mp4"
            if not video_path.is_file():
                return None
            try:
                self._video_readers[key] = VideoReader(str(video_path), ctx=cpu(0))
            except Exception:
                return None
        return self._video_readers[key]

    def _load_frame(self, vr, frame_idx: int) -> np.ndarray:
        if vr is None:
            return np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        frame_idx = max(0, min(frame_idx, len(vr) - 1))
        frame = vr[frame_idx].asnumpy()  # H, W, C RGB
        if (frame.shape[0], frame.shape[1]) != self.image_size:
            frame = cv2.resize(frame, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_AREA)
        return frame

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_idx, frame_idx, ep_len = self.samples[idx]
        ep_data = self._get_parquet_data(ep_idx)

        # 1. State
        raw_state = ep_data["state"][frame_idx]
        if self.state_mean is not None:
            norm_state = (raw_state - self.state_mean) / self.state_std
        else:
            norm_state = raw_state
        state_tensor = torch.tensor(norm_state, dtype=torch.float32)

        # 2. Action chunk [t : t + chunk_size]
        actions = []
        for step in range(self.chunk_size):
            act_idx = min(frame_idx + step, ep_len - 1)
            raw_action = ep_data["action"][act_idx]
            if self.action_mean is not None:
                norm_action = (raw_action - self.action_mean) / self.action_std
            else:
                norm_action = raw_action
            actions.append(norm_action)
        action_tensor = torch.tensor(np.stack(actions), dtype=torch.float32)

        # 3. Task text
        task_idx = int(ep_data["task_index"][frame_idx])
        task_text = self.tasks.get(task_idx, "robot manipulation task")

        # 4. Images: 3 timestamps [t-1, t, t_future]
        t_prev = max(0, frame_idx - 1)
        t_curr = frame_idx
        t_future = min(frame_idx + self.future_horizon, ep_len - 1)

        raw_sample = {
            "task": task_text,
            "observation.state": state_tensor,
            "action": action_tensor,
        }

        # Canonical slots: image0, image1, image2
        canonical_slots = ["image0", "image1", "image2"]
        for slot in canonical_slots:
            raw_sample[f"{OBS_IMAGES}.{slot}_mask"] = torch.tensor(False)
            raw_sample[f"{OBS_IMAGES}.{slot}"] = torch.zeros((3, 3, *self.image_size), dtype=torch.float32)

        for src_cam, target_slot in self.camera_map.items():
            vr = self._get_video_reader(src_cam, ep_idx)
            f_prev = self._load_frame(vr, t_prev)
            f_curr = self._load_frame(vr, t_curr)
            f_future = self._load_frame(vr, t_future)

            # Stack to (3, C, H, W) normalized to [0, 1]
            img_stack = np.stack([f_prev, f_curr, f_future], axis=0)  # (3, H, W, C)
            img_tensor = torch.from_numpy(img_stack).permute(0, 3, 1, 2).float() / 255.0

            raw_sample[f"{OBS_IMAGES}.{target_slot}"] = img_tensor
            raw_sample[f"{OBS_IMAGES}.{target_slot}_mask"] = torch.tensor(True)

        # 5. Apply Processor (produces pixel_values, image_grid_thw, input_ids, attention_mask)
        processed = self.processor(raw_sample)
        return processed


def collate_pelican_batch(batch: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Custom collate function for Pelican-VLA batches."""
    collated = {}

    # Simple stack tensors
    stack_keys = [
        "observation.images.image0",
        "observation.images.image0_mask",
        "observation.images.image1",
        "observation.images.image1_mask",
        "observation.images.image2",
        "observation.images.image2_mask",
        "observation.state",
        "action",
        "observation.input_ids",
        "observation.attention_mask",
        "observation.task.input_ids",
        "observation.task.attention_mask",
    ]

    for k in stack_keys:
        if k in batch[0]:
            collated[k] = torch.stack([b[k] for b in batch], dim=0)

    # Cat tokens for variable-length/patch lists if needed
    if "observation.pixel_values" in batch[0]:
        collated["observation.pixel_values"] = torch.cat([b["observation.pixel_values"] for b in batch], dim=0)
    if "observation.image_grid_thw" in batch[0]:
        collated["observation.image_grid_thw"] = torch.cat([b["observation.image_grid_thw"] for b in batch], dim=0)

    return collated
