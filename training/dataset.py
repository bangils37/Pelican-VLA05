"""Dataset loader for Pelican-VLA 0.5 compatible with LeRobot v2.1 and v3.0 datasets.

Supports:
  - Episode-level train/validation split (preventing data leakage across frames)
  - LeRobot v3.0 chunked video decoding
  - Subtask-conditioned language instructions
  - In-memory tensor caching for high-throughput training
"""

from __future__ import annotations

import json
from pathlib import Path
import random
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
    """Dataset for training Pelican-VLA 0.5 on LeRobot v2.1/v3.0 formatted datasets.
    
    Extracts:
      - 3 temporal image frames per camera: [t-delta, t, t+delta] (T=3, delta=15 frames)
      - State at current frame t: [32] (normalized and padded to 32 dimensions)
      - Action chunk from t to t + chunk_size: [chunk_size, 32] (normalized and padded to 32 dimensions)
      - Task / subtask instruction text
      - Normalized state and action using dataset stats
      - Episode-level train/validation splitting
    """

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        camera_map: Mapping[str, str] | None = None,
        chunk_size: int = 50,
        future_horizon: int = 15,  # frame index delta for future ground truth target (t+15)
        qwen3_vl_path: str = "Qwen/Qwen3-VL-4B-Instruct",
        max_length: int = 48,
        image_size: tuple[int, int] = (224, 224),
        pad_dim: int = 32,
        use_subtasks_prob: float = 0.5,
        split: str = "train",  # "train", "val", or "all"
        val_ratio: float = 0.05,  # Ratio of held-out episodes for validation
        seed: int = 42,
        _shared_data: dict | None = None,
    ):
        super().__init__()
        self.root = Path(dataset_root).expanduser().resolve()
        self.chunk_size = chunk_size
        self.future_horizon = future_horizon
        self.image_size = image_size
        self.pad_dim = pad_dim
        self.use_subtasks_prob = use_subtasks_prob
        self.split = split.lower()

        # Load info.json
        info_path = self.root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"info.json not found at {info_path}")
        with open(info_path, "r", encoding="utf-8") as f:
            self.info = json.load(f)

        self.fps = self.info.get("fps", 30)

        # Normalize camera_map
        if camera_map is None:
            camera_map = {
                "cam_head": "image0",
                "cam_left_wrist": "image1",
                "cam_right_wrist": "image2",
            }
        
        # Build flexible mapping allowing prefix 'observation.images.' or raw camera names
        self.camera_map = {}
        for src_cam, target_slot in camera_map.items():
            clean_name = src_cam.removeprefix("observation.images.")
            self.camera_map[clean_name] = target_slot

        # Load tasks
        self.tasks = {}
        tasks_parquet = self.root / "meta" / "tasks.parquet"
        tasks_jsonl = self.root / "meta" / "tasks.jsonl"
        if tasks_parquet.is_file():
            df_tasks = pq.read_table(tasks_parquet).to_pandas().reset_index()
            task_col = "task" if "task" in df_tasks.columns else df_tasks.columns[0]
            idx_col = "task_index" if "task_index" in df_tasks.columns else df_tasks.columns[1]
            for _, row in df_tasks.iterrows():
                self.tasks[int(row[idx_col])] = str(row[task_col])
        elif tasks_jsonl.is_file():
            with open(tasks_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line.strip())
                    self.tasks[item["task_index"]] = item["task"]

        # Load subtasks if available
        self.subtasks = {}
        subtasks_parquet = self.root / "meta" / "subtasks.parquet"
        if subtasks_parquet.is_file():
            df_sub = pq.read_table(subtasks_parquet).to_pandas().reset_index()
            sub_col = "subtask" if "subtask" in df_sub.columns else df_sub.columns[0]
            idx_col = "subtask_index" if "subtask_index" in df_sub.columns else df_sub.columns[1]
            for _, row in df_sub.iterrows():
                self.subtasks[int(row[idx_col])] = str(row[sub_col])

        # Load stats
        self.state_mean = None
        self.state_std = None
        self.action_mean = None
        self.action_std = None
        self._load_stats()

        # Load episode metadata
        self.episodes = []
        episodes_parquet = self.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        episodes_jsonl = self.root / "meta" / "episodes.jsonl"
        if episodes_parquet.is_file():
            self._is_v3 = True
            ep_df = pq.read_table(episodes_parquet).to_pandas()
            for _, row in ep_df.iterrows():
                ep_dict = row.to_dict()
                self.episodes.append(ep_dict)
        elif episodes_jsonl.is_file():
            self._is_v3 = False
            with open(episodes_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    self.episodes.append(json.loads(line.strip()))
        else:
            raise FileNotFoundError(f"Neither {episodes_parquet} nor {episodes_jsonl} was found.")

        # Episode-level train/validation split
        total_eps = len(self.episodes)
        if val_ratio > 0.0 and self.split in ["train", "val"]:
            rng = np.random.RandomState(seed)
            shuffled_eps = rng.permutation(total_eps).tolist()
            n_val = max(1, int(round(total_eps * val_ratio)))
            val_eps_set = set(shuffled_eps[:n_val])
            train_eps_set = set(shuffled_eps[n_val:])
            self.active_episodes_set = val_eps_set if self.split == "val" else train_eps_set
        else:
            self.active_episodes_set = set(range(total_eps))

        # Load primary data parquet table into memory (or reuse shared cache)
        if _shared_data is not None:
            self.raw_states = _shared_data["raw_states"]
            self.raw_actions = _shared_data["raw_actions"]
            self.task_indices = _shared_data["task_indices"]
            self.subtask_indices = _shared_data["subtask_indices"]
            self._use_in_memory_data = True
        else:
            data_parquet_v3 = self.root / "data" / "chunk-000" / "file-000.parquet"
            if data_parquet_v3.is_file():
                data_tbl = pq.read_table(
                    data_parquet_v3,
                    columns=["observation.state", "action", "task_index", "subtask_index"]
                    if "subtask_index" in pq.read_schema(data_parquet_v3).names
                    else ["observation.state", "action", "task_index"],
                )
                self.raw_states = np.stack(data_tbl["observation.state"].to_numpy()).astype(np.float32)
                self.raw_actions = np.stack(data_tbl["action"].to_numpy()).astype(np.float32)
                self.task_indices = data_tbl["task_index"].to_numpy().astype(np.int64)
                if "subtask_index" in data_tbl.column_names:
                    self.subtask_indices = data_tbl["subtask_index"].to_numpy().astype(np.int64)
                else:
                    self.subtask_indices = None
                self._use_in_memory_data = True
            else:
                self._use_in_memory_data = False
                self._parquet_cache = {}

        # Build index map: list of (global_row_idx, episode_idx, frame_in_ep, ep_len, ep_start_idx)
        self.samples = []
        for ep in self.episodes:
            ep_idx = int(ep["episode_index"])
            if ep_idx not in self.active_episodes_set:
                continue
            ep_len = int(ep["length"])
            ep_start_idx = int(ep.get("dataset_from_index", 0))
            for frame_idx in range(ep_len):
                global_row_idx = ep_start_idx + frame_idx
                self.samples.append((global_row_idx, ep_idx, frame_idx, ep_len, ep_start_idx))

        # Initialize processor transform
        self.processor = PelicanVLA05ProcessorTransformFn(
            pretrained_model_name_or_path=qwen3_vl_path,
            max_length=max_length,
        )

        # Cache open video readers: key=(cam_name, file_idx) -> VideoReader
        self._video_readers = {}

    def get_shared_data(self) -> dict:
        """Export in-memory arrays to quickly initialize validation dataset without re-reading disks."""
        if not self._use_in_memory_data:
            return {}
        return {
            "raw_states": self.raw_states,
            "raw_actions": self.raw_actions,
            "task_indices": self.task_indices,
            "subtask_indices": self.subtask_indices,
        }

    def _load_stats(self):
        stats_json = self.root / "meta" / "stats.json"
        stats_jsonl = self.root / "meta" / "episodes_stats.jsonl"
        if stats_json.is_file():
            with open(stats_json, "r", encoding="utf-8") as f:
                stats = json.load(f)
            if "observation.state" in stats:
                self.state_mean = np.array(stats["observation.state"]["mean"], dtype=np.float32)
                self.state_std = np.array(stats["observation.state"]["std"], dtype=np.float32)
                self.state_std[self.state_std < 1e-4] = 1.0
            if "action" in stats:
                self.action_mean = np.array(stats["action"]["mean"], dtype=np.float32)
                self.action_std = np.array(stats["action"]["std"], dtype=np.float32)
                self.action_std[self.action_std < 1e-4] = 1.0
        elif stats_jsonl.is_file():
            with open(stats_jsonl, "r", encoding="utf-8") as f:
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

    def _get_video_reader(self, cam_name: str, ep_dict: dict):
        if self._is_v3:
            cam_key = f"observation.images.{cam_name}"
            file_idx_key = f"videos/{cam_key}/file_index"
            if file_idx_key not in ep_dict:
                file_idx_key = f"videos/{cam_name}/file_index"
            
            file_idx = int(ep_dict.get(file_idx_key, 0))
            cache_key = (cam_name, file_idx)

            if cache_key not in self._video_readers:
                video_path = self.root / "videos" / cam_key / "chunk-000" / f"file-{file_idx:03d}.mp4"
                if not video_path.is_file():
                    video_path = self.root / "videos" / cam_name / "chunk-000" / f"file-{file_idx:03d}.mp4"
                if not video_path.is_file():
                    return None, 0
                try:
                    self._video_readers[cache_key] = VideoReader(str(video_path), ctx=cpu(0))
                except Exception:
                    return None, 0

            from_ts_key = f"videos/{cam_key}/from_timestamp"
            if from_ts_key not in ep_dict:
                from_ts_key = f"videos/{cam_name}/from_timestamp"
            from_ts = float(ep_dict.get(from_ts_key, 0.0))
            start_frame = int(round(from_ts * self.fps))
            return self._video_readers[cache_key], start_frame
        else:
            ep_idx = int(ep_dict["episode_index"])
            cache_key = (cam_name, ep_idx)
            if cache_key not in self._video_readers:
                video_path = self.root / "videos" / "chunk-000" / f"observation.images.{cam_name}" / f"episode_{ep_idx:06d}.mp4"
                if not video_path.is_file():
                    video_path = self.root / "videos" / "chunk-000" / cam_name / f"episode_{ep_idx:06d}.mp4"
                if not video_path.is_file():
                    return None, 0
                try:
                    self._video_readers[cache_key] = VideoReader(str(video_path), ctx=cpu(0))
                except Exception:
                    return None, 0
            return self._video_readers[cache_key], 0

    def _load_frame(self, vr: VideoReader | None, frame_idx: int) -> np.ndarray:
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
        global_row_idx, ep_idx, frame_in_ep, ep_len, ep_start_idx = self.samples[idx]
        ep_dict = self.episodes[ep_idx]

        # 1. State
        if self._use_in_memory_data:
            raw_state = self.raw_states[global_row_idx]
            task_idx = int(self.task_indices[global_row_idx])
            subtask_idx = int(self.subtask_indices[global_row_idx]) if self.subtask_indices is not None else -1
        else:
            if ep_idx not in self._parquet_cache:
                parquet_path = self.root / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
                table = pq.read_table(parquet_path)
                self._parquet_cache[ep_idx] = {
                    "state": np.stack(table["observation.state"].to_numpy()),
                    "action": np.stack(table["action"].to_numpy()),
                    "task_index": table["task_index"].to_numpy(),
                }
            ep_data = self._parquet_cache[ep_idx]
            raw_state = ep_data["state"][frame_in_ep]
            task_idx = int(ep_data["task_index"][frame_in_ep])
            subtask_idx = -1

        if self.state_mean is not None:
            norm_state = (raw_state - self.state_mean) / self.state_std
        else:
            norm_state = raw_state

        # Zero-pad state to pad_dim (32)
        padded_state = np.zeros(self.pad_dim, dtype=np.float32)
        orig_s_dim = min(len(norm_state), self.pad_dim)
        padded_state[:orig_s_dim] = norm_state[:orig_s_dim]
        state_tensor = torch.from_numpy(padded_state).float()

        # 2. Action chunk [t : t + chunk_size]
        actions = []
        ep_end_row = ep_start_idx + ep_len - 1
        for step in range(self.chunk_size):
            act_row = min(global_row_idx + step, ep_end_row)
            if self._use_in_memory_data:
                raw_act = self.raw_actions[act_row]
            else:
                act_idx = min(frame_in_ep + step, ep_len - 1)
                raw_act = ep_data["action"][act_idx]

            if self.action_mean is not None:
                norm_act = (raw_act - self.action_mean) / self.action_std
            else:
                norm_act = raw_act

            # Zero-pad action to pad_dim (32)
            padded_act = np.zeros(self.pad_dim, dtype=np.float32)
            orig_a_dim = min(len(norm_act), self.pad_dim)
            padded_act[:orig_a_dim] = norm_act[:orig_a_dim]
            actions.append(padded_act)

        action_tensor = torch.from_numpy(np.stack(actions)).float()

        # 3. Task text
        task_text = self.tasks.get(task_idx, "serving brewed coffee")
        if subtask_idx >= 0 and subtask_idx in self.subtasks:
            if random.random() < self.use_subtasks_prob:
                task_text = self.subtasks[subtask_idx]

        # 4. Images: 3 timestamps [t-delta, t, t+delta] (T=3)
        t_prev_rel = max(0, frame_in_ep - self.future_horizon)
        t_curr_rel = frame_in_ep
        t_future_rel = min(frame_in_ep + self.future_horizon, ep_len - 1)

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
            vr, start_f = self._get_video_reader(src_cam, ep_dict)
            f_prev = self._load_frame(vr, start_f + t_prev_rel)
            f_curr = self._load_frame(vr, start_f + t_curr_rel)
            f_future = self._load_frame(vr, start_f + t_future_rel)

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
