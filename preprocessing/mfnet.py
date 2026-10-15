"""
CTCM-Net: Production MFNet Dataset Pipeline (Steps 1 to 5) - Real Data Ingestion
================================================================================
Module: preprocessing/mfnet.py
Hardware Target: NVIDIA RTX A6000

The 5-Step Pipeline Implementation for MFNet:
1. Step 1: Spatial Planar Homography Alignment to correct camera parallax between RGB and Thermal.
2. Step 2: Strict [256, 256] Resize & 9-class discrete integer mask via NEAREST interpolation.
3. Step 3: Aligned data augmentations: identical geometric transforms for RGB, Thermal, and mask.
4. Step 4: Train-set normalization using strictly MFNet training set statistics.
5. Step 5: Dataset splits (train/val/test) wrapped in PyTorch DataLoader (num_workers=8, pin_memory=True).

NO DUMMY / SYNTHETIC FALLBACKS: Real dataset files must exist on disk.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from preprocessing.homography import SpatialPlanarHomographyAlignment
from preprocessing.target_formulation import encode_mfnet_mask, MFNET_CLASSES, NUM_SEG_CLASSES
from preprocessing.transforms import AlignedMultimodalTransform
from preprocessing.normalization import TrainSetNormalizer


def resolve_mfnet_root(root_dir: str = "Datasets/MFNet/ir_seg_dataset") -> str:
    """Recursively locates the MFNet dataset root directory or raises FileNotFoundError."""
    candidates = [
        root_dir,
        os.path.abspath(root_dir),
        os.path.join(".", root_dir),
        os.path.join("/home/rounak/CTCM-Net", root_dir.lstrip("./")),
        os.path.join("/home/rounak/CTCM-Net/Datasets/MFNet/ir_seg_dataset"),
        os.path.join("/home/rounak/CTCM-Net/datasets/MFNet/ir_seg_dataset"),
        "Datasets/MFNet/ir_seg_dataset",
        "datasets/MFNet/ir_seg_dataset",
        "ir_seg_dataset"
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.isdir(c):
            # Verify images and labels directories exist inside
            if os.path.exists(os.path.join(c, "images")) and os.path.exists(os.path.join(c, "labels")):
                return os.path.abspath(c)

    raise FileNotFoundError(
        f"CRITICAL ERROR: MFNet dataset directory not found!\n"
        f"Searched candidates:\n" + "\n".join(f" - {c}" for c in candidates) + "\n"
        f"Please verify that the dataset is present under ./Datasets/MFNet/ir_seg_dataset."
    )


class MFNetDataset(Dataset):
    """
    Production-grade PyTorch Dataset for MFNet RGB-Thermal Urban Semantic Segmentation.
    Strictly ingests REAL empirical dataset files.
    """

    def __init__(
        self,
        root_dir: str = "Datasets/MFNet/ir_seg_dataset",
        split: str = "train",
        target_size: Tuple[int, int] = (256, 256),
        is_train: Optional[bool] = None,
        apply_homography: bool = True,
        augment: bool = True,
        normalize: bool = True
    ):
        super().__init__()
        self.root_dir = resolve_mfnet_root(root_dir)
        self.split = split.lower()
        self.target_size = target_size
        self.is_train = (self.split == "train") if is_train is None else is_train
        self.apply_homography = apply_homography
        self.augment = augment
        self.normalize = normalize

        # Step 1: Homography Registration Module
        self.homography_module = SpatialPlanarHomographyAlignment()

        # Step 3: Aligned Multimodal Transformations
        self.transform = AlignedMultimodalTransform(
            output_size=target_size,
            crop_scale=(0.8, 1.0) if (self.is_train and augment) else (1.0, 1.0),
            p_flip=0.5 if (self.is_train and augment) else 0.0,
            is_train=self.is_train and augment
        )

        # Step 4: Train-set Statistical Normalizer
        self.normalizer = TrainSetNormalizer(dataset_name="MFNet")

        # Discover real sample list
        self.samples = self._load_split_samples()

    def _load_split_samples(self) -> List[Dict[str, Any]]:
        """Loads real sample metadata from split file or raises FileNotFoundError."""
        split_file = os.path.join(self.root_dir, f"{self.split}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"MFNet split file '{split_file}' not found! "
                f"Expected train.txt, val.txt, or test.txt inside {self.root_dir}."
            )

        with open(split_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        samples: List[Dict[str, Any]] = []
        for line in lines:
            is_flip = line.endswith("_flip")
            base_name = line[:-5] if is_flip else line
            img_path = os.path.join(self.root_dir, "images", f"{base_name}.png")
            lbl_path = os.path.join(self.root_dir, "labels", f"{base_name}.png")

            if not os.path.exists(img_path):
                raise FileNotFoundError(f"MFNet image missing: {img_path}")
            if not os.path.exists(lbl_path):
                raise FileNotFoundError(f"MFNet label missing: {lbl_path}")

            samples.append({
                "name": line,
                "base_name": base_name,
                "img_path": img_path,
                "lbl_path": lbl_path,
                "is_flip": is_flip
            })

        if len(samples) == 0:
            raise FileNotFoundError(f"MFNet split '{self.split}' contained zero valid samples in {split_file}!")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Load real 4-channel RGBA image: channels 0..2 = RGB, channel 3 = LWIR Thermal
        pil_img = Image.open(sample["img_path"])
        arr_img = np.array(pil_img, dtype=np.float32) / 255.0  # [H, W, 4]
        rgb_raw = torch.from_numpy(arr_img[:, :, :3]).permute(2, 0, 1).float()  # [3, H, W]
        th_1ch = torch.from_numpy(arr_img[:, :, 3:4]).permute(2, 0, 1).float()   # [1, H, W]
        thermal_raw = th_1ch.repeat(3, 1, 1)  # Replicated to 3 channels for vision backbones

        # Load real label mask (discrete integers 0..8)
        pil_lbl = Image.open(sample["lbl_path"])
        mask_raw = torch.from_numpy(np.array(pil_lbl, dtype=np.int64)).long()  # [H, W]

        # Apply horizontal flip if specified by split list
        if sample["is_flip"]:
            rgb_raw = torch.flip(rgb_raw, dims=[2])
            thermal_raw = torch.flip(thermal_raw, dims=[2])
            mask_raw = torch.flip(mask_raw, dims=[1])

        # STEP 1: Spatial Planar Homography Alignment (Parallax Correction)
        if self.apply_homography:
            aligned_rgb, aligned_thermal, homography_matrix = self.homography_module(
                rgb_raw, thermal_raw
            )
        else:
            aligned_rgb, aligned_thermal = rgb_raw, thermal_raw
            homography_matrix = self.homography_module.get_calibrated_homography()

        # STEP 2: Strict NEAREST-neighbor Mask Encoding to [256, 256]
        mask_encoded = encode_mfnet_mask(mask_raw, target_size=self.target_size)

        # STEP 3: Aligned Multimodal Transformations (identical geometry across RGB, T, Mask)
        if self.is_train and self.augment:
            trans_rgb, trans_thermal, trans_mask = self.transform(
                aligned_rgb,
                aligned_thermal,
                target=mask_encoded,
                is_density=False
            )
        else:
            # Deterministic resize to [256, 256] in eval or unaugmented mode
            import torchvision.transforms.functional as TF
            trans_rgb = TF.resize(aligned_rgb, list(self.target_size), interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
            trans_thermal = TF.resize(aligned_thermal, list(self.target_size), interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
            trans_mask = mask_encoded

        # STEP 4: Train-Set Statistical Normalization
        if self.normalize:
            norm_rgb, norm_thermal = self.normalizer(trans_rgb, trans_thermal)
        else:
            norm_rgb, norm_thermal = trans_rgb, trans_thermal

        return {
            "rgb": norm_rgb.float(),                        # [3, 256, 256] normalized
            "thermal": norm_thermal.float(),                # [3, 256, 256] normalized
            "rgb_unnorm": trans_rgb.float(),                # [3, 256, 256] in [0, 1]
            "thermal_unnorm": trans_thermal.float(),        # [3, 256, 256] in [0, 1]
            "sem_seg_mask": trans_mask.long(),              # [256, 256] (discrete integer tokens 0..8)
            "homography_matrix": homography_matrix.float(), # [3, 3]
            "sample_name": sample["name"],
            "dataset": "MFNet"
        }


def build_mfnet_dataloader(
    root_dir: str = "Datasets/MFNet/ir_seg_dataset",
    split: str = "train",
    batch_size: int = 8,
    num_workers: int = 8,
    pin_memory: bool = True,
    shuffle: Optional[bool] = None
) -> DataLoader:
    """
    Step 5: Wrap MFNet dataset in PyTorch DataLoader optimized for NVIDIA RTX A6000.
    """
    is_train = (split.lower() == "train")
    if shuffle is None:
        shuffle = is_train

    dataset = MFNetDataset(
        root_dir=root_dir,
        split=split,
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
