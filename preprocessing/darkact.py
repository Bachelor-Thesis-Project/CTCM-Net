"""
CTCM-Net: Production DarkAct Dataset Pipeline (Steps 1 to 5) - Real Data Ingestion
==================================================================================
Module: preprocessing/darkact.py
Hardware Target: NVIDIA RTX A6000

The 5-Step Pipeline Implementation for DarkAct:
1. Step 1: Hardware-level temporal genlock sampling (16 consecutive frames at 30 FPS / 33.3ms intervals).
2. Step 2: Strict [256, 256] spatial frame resize & 27 discrete action class logits [B, 27].
3. Step 3: Aligned video transformations: identical spatial transforms across all 16 frames simultaneously.
4. Step 4: Train-set normalization using strictly DarkAct training set statistics.
5. Step 5: Dataset splits (train/val/test) partitioned by non-overlapping subjects/scenes, wrapped in DataLoader.

NO DUMMY / SYNTHETIC FALLBACKS: Real dataset files must exist on disk.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader

from preprocessing.temporal_sync import HardwareTemporalGenlockSampler
from preprocessing.target_formulation import encode_darkact_action, DARKACT_CLASSES, NUM_ACTION_CLASSES
from preprocessing.transforms import AlignedMultimodalTransform
from preprocessing.normalization import TrainSetNormalizer


def resolve_darkact_root(root_dir: str = "Datasets/DarkAct/ARIM_v1") -> str:
    """Recursively locates the DarkAct dataset root directory or raises FileNotFoundError."""
    candidates = [
        root_dir,
        os.path.abspath(root_dir),
        os.path.join(".", root_dir),
        os.path.join("/home/rounak/CTCM-Net", root_dir.lstrip("./")),
        os.path.join("/home/rounak/CTCM-Net/Datasets/DarkAct/ARIM_v1"),
        os.path.join("/home/rounak/CTCM-Net/datasets/DarkAct/ARIM_v1"),
        "Datasets/DarkAct/ARIM_v1",
        "datasets/DarkAct/ARIM_v1",
        "ARIM_v1"
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.isdir(c):
            # Verify annotations and videos directories exist inside
            if os.path.exists(os.path.join(c, "annotations")) and os.path.exists(os.path.join(c, "videos")):
                return os.path.abspath(c)

    raise FileNotFoundError(
        f"CRITICAL ERROR: DarkAct dataset directory not found!\n"
        f"Searched candidates:\n" + "\n".join(f" - {c}" for c in candidates) + "\n"
        f"Please verify that the dataset is present under ./Datasets/DarkAct/ARIM_v1."
    )


class DarkActDataset(Dataset):
    """
    Production-grade PyTorch Dataset for DarkAct Night Action Recognition.
    Strictly ingests REAL empirical dataset video clips.
    """

    def __init__(
        self,
        root_dir: str = "Datasets/DarkAct/ARIM_v1",
        split: str = "train",
        sequence_length: int = 16,
        target_size: Tuple[int, int] = (256, 256),
        is_train: Optional[bool] = None,
        augment: bool = True,
        normalize: bool = True,
        val_ratio: float = 0.15,
        seed: int = 42
    ):
        super().__init__()
        self.root_dir = resolve_darkact_root(root_dir)
        self.split = split.lower()
        self.sequence_length = sequence_length
        self.target_size = target_size
        self.is_train = (self.split == "train") if is_train is None else is_train
        self.augment = augment
        self.normalize = normalize
        self.val_ratio = val_ratio
        self.seed = seed

        # Step 1: Hardware Temporal Genlock Sampler (consecutive 16 frames at 30 FPS / 33.3ms intervals)
        self.genlock_sampler = HardwareTemporalGenlockSampler(
            sequence_length=sequence_length,
            target_fps=30.0,
            target_size=target_size
        )

        # Step 3: Aligned Video Transformations (applies identical spatial transforms to all 16 frames)
        self.transform = AlignedMultimodalTransform(
            output_size=target_size,
            crop_scale=(0.85, 1.0) if (self.is_train and augment) else (1.0, 1.0),
            p_flip=0.5 if (self.is_train and augment) else 0.0,
            is_train=self.is_train and augment
        )

        # Step 4: Train-set Statistical Normalizer
        self.normalizer = TrainSetNormalizer(dataset_name="DarkAct")

        # Discover real sample list with zero-leakage subject partitioning
        self.samples = self._load_split_samples()

    def _load_split_samples(self) -> List[Dict[str, Any]]:
        """
        Loads real sample metadata with strict non-overlapping subject/session partitioning.
        """
        anno_dir = os.path.join(self.root_dir, "annotations")
        train_anno = os.path.join(anno_dir, "_multimodal_train_annotations.txt")
        test_anno = os.path.join(anno_dir, "_multimodal_test_annotations.txt")

        if not os.path.exists(train_anno):
            raise FileNotFoundError(f"Missing DarkAct train annotation file: {train_anno}")
        if not os.path.exists(test_anno):
            raise FileNotFoundError(f"Missing DarkAct test annotation file: {test_anno}")

        samples: List[Dict[str, Any]] = []

        if self.split == "test":
            with open(test_anno, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        rgb_p = os.path.join(self.root_dir, "videos", parts[0])
                        th_p = os.path.join(self.root_dir, "videos", parts[1])
                        label = int(parts[2])
                        if not os.path.exists(rgb_p):
                            raise FileNotFoundError(f"Missing RGB video file: {rgb_p}")
                        if not os.path.exists(th_p):
                            raise FileNotFoundError(f"Missing Thermal video file: {th_p}")
                        samples.append({
                            "rgb_path": rgb_p,
                            "th_path": th_p,
                            "label": label
                        })
        else:
            with open(train_anno, "r") as f:
                raw_lines = [l.strip().split() for l in f if l.strip()]

            # Group by subject / session prefix to guarantee zero leakage between train & val
            session_groups: Dict[str, List[Tuple[str, str, int]]] = {}
            for parts in raw_lines:
                if len(parts) >= 3:
                    rgb_rel = parts[0]
                    th_rel = parts[1]
                    label = int(parts[2])
                    base = os.path.basename(rgb_rel)
                    session_key = base.split("_rgb")[0] if "_rgb" in base else base
                    if session_key not in session_groups:
                        session_groups[session_key] = []

                    rgb_p = os.path.join(self.root_dir, "videos", rgb_rel)
                    th_p = os.path.join(self.root_dir, "videos", th_rel)
                    if not os.path.exists(rgb_p):
                        raise FileNotFoundError(f"Missing RGB video file: {rgb_p}")
                    if not os.path.exists(th_p):
                        raise FileNotFoundError(f"Missing Thermal video file: {th_p}")
                    session_groups[session_key].append((rgb_p, th_p, label))

            all_sessions = sorted(list(session_groups.keys()))
            num_val_sessions = max(1, int(len(all_sessions) * self.val_ratio))

            import random
            rng = random.Random(self.seed)
            shuffled_sessions = list(all_sessions)
            rng.shuffle(shuffled_sessions)

            val_session_set = set(shuffled_sessions[:num_val_sessions])
            train_session_set = set(shuffled_sessions[num_val_sessions:])

            target_set = val_session_set if self.split == "val" else train_session_set

            for s_key in target_set:
                for rgb_p, th_p, label in session_groups[s_key]:
                    samples.append({
                        "rgb_path": rgb_p,
                        "th_path": th_p,
                        "label": label
                    })

        if len(samples) == 0:
            raise FileNotFoundError(f"DarkAct split '{self.split}' contained zero valid clips!")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        action_label = sample["label"]

        # STEP 1: Hardware-Level Temporal Genlock Sampling (16 consecutive frames at 30 FPS / 33.3ms intervals)
        rgb_seq, th_seq, timestamps = self.genlock_sampler.sample_video_pair(
            rgb_video_path=sample["rgb_path"],
            thermal_video_path=sample["th_path"],
            is_train=self.is_train
        )

        # STEP 2: Strict [256, 256] Frame Resize & Action Target Formulation
        import torchvision.transforms.functional as TF
        rgb_seq_256 = TF.resize(rgb_seq, list(self.target_size), interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        th_seq_256 = TF.resize(th_seq, list(self.target_size), interpolation=TF.InterpolationMode.BILINEAR, antialias=True)

        action_logits, action_idx = encode_darkact_action(
            action_label=action_label,
            num_classes=NUM_ACTION_CLASSES,
            label_smoothing=0.0
        )

        # STEP 3: Aligned Video Transformations (identically across all 16 frames)
        if self.is_train and self.augment:
            trans_rgb, trans_thermal, _ = self.transform(
                rgb_seq_256,
                th_seq_256,
                target=None
            )
        else:
            trans_rgb, trans_thermal = rgb_seq_256, th_seq_256

        # STEP 4: Train-Set Statistical Normalization
        if self.normalize:
            norm_rgb, norm_thermal = self.normalizer(trans_rgb, trans_thermal)
        else:
            norm_rgb, norm_thermal = trans_rgb, trans_thermal

        return {
            "rgb": norm_rgb.float(),                        # [16, 3, 256, 256] normalized
            "thermal": norm_thermal.float(),                # [16, 3, 256, 256] normalized
            "rgb_unnorm": trans_rgb.float(),                # [16, 3, 256, 256] in [0, 1]
            "thermal_unnorm": trans_thermal.float(),        # [16, 3, 256, 256] in [0, 1]
            "action_logits": action_logits.float(),         # [27] discrete action class logits
            "action_label": action_idx.long(),              # scalar [0 to 26]
            "timestamps_ms": torch.tensor(timestamps, dtype=torch.float32),  # [16] (at 33.3ms intervals)
            "action_name": DARKACT_CLASSES[action_label],
            "dataset": "DarkAct"
        }


def build_darkact_dataloader(
    root_dir: str = "Datasets/DarkAct/ARIM_v1",
    split: str = "train",
    sequence_length: int = 16,
    batch_size: int = 4,
    num_workers: int = 8,
    pin_memory: bool = True,
    shuffle: Optional[bool] = None
) -> DataLoader:
    """
    Step 5: Wrap DarkAct dataset in PyTorch DataLoader optimized for NVIDIA RTX A6000.
    """
    is_train = (split.lower() == "train")
    if shuffle is None:
        shuffle = is_train

    dataset = DarkActDataset(
        root_dir=root_dir,
        split=split,
        sequence_length=sequence_length,
        target_size=(256, 256),
        is_train=is_train
    )

    use_pin = pin_memory and torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_pin,
        drop_last=is_train
    )
