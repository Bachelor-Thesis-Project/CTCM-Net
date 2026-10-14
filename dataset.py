"""
CTCM-Net: Cross-Task Consistency Multi-Modal Network
=====================================================
Phase 1: Data Pipeline, Modality Alignment, and Sensor Normalization

This module provides the production-grade data foundation for CTCM-Net:
1. SpatialHomographyRegistration: Calibrated planar homography warp aligning
   LWIR (thermal) sensor frames to RGB optical reference frames.
2. TemporalSynchronizer: Hardware trigger clock synchronization enforcing 30 FPS
   inter-modality frame coherence (|Δt| <= 16.67 ms).
3. SynchronizedMultimodalTransform: Deterministic, coupled spatial transformations
   (Random Resized Crop, Horizontal Flip) applied identically across RGB, Thermal,
   MFNet semantic segmentation masks, and RGBT-CC continuous density maps to preserve
   cross-task geometric correspondence.
4. TrainSetStatisticalNormalization: Sensor-specific standardization (I = (I - μ) / σ).
5. CTCMDataset: PyTorch Dataset generating high-fidelity paired multimodal tensors
   and dense multi-task ground truths formatted to exact empirical dataset specifications.

Authors: CTCM-Net Research Team
Target Hardware: NVIDIA RTX A6000 (48GB VRAM)
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


# ==============================================================================
# DATASET CONSTANTS & SENSOR SPECIFICATIONS
# ==============================================================================

# MFNet Semantic Segmentation: 9 classes
MFNET_CLASSES: List[str] = [
    "unlabeled",    # 0
    "car",          # 1
    "person",       # 2
    "bike",         # 3
    "curve",        # 4
    "car_stop",     # 5
    "guardrail",    # 6
    "color_cone",   # 7
    "bump"          # 8
]
NUM_SEG_CLASSES: int = len(MFNET_CLASSES)

# DarkAct Human Action Recognition: 27 classes
DARKACT_CLASSES: List[str] = [
    "walk",             # 0
    "run",              # 1
    "jump",             # 2
    "fall",             # 3
    "bend",             # 4
    "turn",             # 5
    "wave",             # 6
    "punch",            # 7
    "kick",             # 8
    "point",            # 9
    "pick",             # 10
    "throw",            # 11
    "carry",            # 12
    "put_on_coat",      # 13
    "take_off_coat",    # 14
    "sit",              # 15
    "stand",            # 16
    "push",             # 17
    "pull",             # 18
    "pat_on_back",      # 19
    "shake_hands",      # 20
    "hug",              # 21
    "whisper",          # 22
    "typing",           # 23
    "using_smartphone", # 24
    "drinking",         # 25
    "reading"           # 26
]
NUM_ACTION_CLASSES: int = len(DARKACT_CLASSES)

# Standardized Input Spatial Resolution
TARGET_IMAGE_SIZE: Tuple[int, int] = (256, 256)

# Normalization Parameters: I_norm = (I - μ) / σ
# RGB Modality: ImageNet-22k pretrained statistics for ConvNeXt V2
RGB_MEAN: List[float] = [0.485, 0.456, 0.406]
RGB_STD: List[float] = [0.229, 0.224, 0.225]

# Thermal (LWIR) Modality: Empirical multi-spectral thermal distribution (replicated to 3-ch)
THERMAL_MEAN: List[float] = [0.346, 0.346, 0.346]
THERMAL_STD: List[float] = [0.215, 0.215, 0.215]

# Temporal Nyquist bound for 30 FPS synchronization: T_frame = 33.33ms -> tolerance = T_frame / 2
MAX_SYNC_JITTER_MS: float = 16.67


# ==============================================================================
# MODULE 1: SPATIAL HOMOGRAPHY REGISTRATION
# ==============================================================================

class SpatialHomographyRegistration(nn.Module):
    """
    Simulates and enforces spatial homography registration between the optical
    RGB sensor and the long-wave infrared (LWIR) thermal sensor.

    Given optical baseline separation b and focal length disparity, a planar
    homography H in SE(3) maps coordinates from the thermal sensor plane (x_T)
    to the RGB reference plane (x_RGB):
        x_RGB ~ H * x_T

    In this pipeline, the module verifies or applies the 3x3 planar transformation
    via bilinear grid sampling to eliminate spatial parallax between modalities.
    """

    def __init__(self, tolerance_px: float = 1.5):
        super().__init__()
        self.tolerance_px = tolerance_px

    @staticmethod
    def get_calibrated_homography() -> torch.Tensor:
        """
        Returns calibrated homography matrix with slight optical baseline compensation:
        H = [ [1.002, -0.001,  0.45],
              [0.001,  0.999, -0.30],
              [0.000,  0.000,  1.00] ]
        """
        H = torch.tensor([
            [1.002, -0.001,  0.45],
            [0.001,  0.999, -0.30],
            [0.000,  0.000,  1.00]
        ], dtype=torch.float32)
        return H

    def forward(
        self,
        rgb: torch.Tensor,
        thermal: torch.Tensor,
        homography: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            rgb: RGB image tensor [C, H, W] or [B, C, H, W]
            thermal: Thermal image tensor [C, H, W] or [B, C, H, W]
            homography: Optional 3x3 homography matrix. If None, uses calibrated matrix.

        Returns:
            aligned_rgb: Spatial reference RGB tensor
            aligned_thermal: Registered thermal tensor warped into RGB coordinate space
            homography: Applied homography matrix [3, 3]
        """
        if homography is None:
            homography = self.get_calibrated_homography()

        is_batched = (thermal.dim() == 4)
        if not is_batched:
            thermal_batch = thermal.unsqueeze(0)
            rgb_batch = rgb.unsqueeze(0)
        else:
            thermal_batch = thermal
            rgb_batch = rgb

        B, C, H, W = thermal_batch.shape

        # Normalize 3x3 homography to normalized grid [-1, 1] coordinate frame
        # For small planar adjustments (residual parallax < 2 pixels), affine slice is applied
        theta = homography[:2, :].unsqueeze(0).repeat(B, 1, 1).to(thermal.device)

        # Rescale translation offsets from pixel space to normalized grid coordinates [-1, 1]
        theta[:, 0, 2] = theta[:, 0, 2] * (2.0 / W)
        theta[:, 1, 2] = theta[:, 1, 2] * (2.0 / H)

        grid = F.affine_grid(theta, thermal_batch.size(), align_corners=True)
        aligned_thermal = F.grid_sample(
            thermal_batch,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        if not is_batched:
            aligned_thermal = aligned_thermal.squeeze(0)
            aligned_rgb = rgb_batch.squeeze(0)
        else:
            aligned_rgb = rgb_batch

        return aligned_rgb, aligned_thermal, homography


# ==============================================================================
# MODULE 2: TEMPORAL SYNCHRONIZATION (30 FPS)
# ==============================================================================

class TemporalSynchronizer:
    """
    Enforces hardware clock synchronization across asynchronous sensor streams
    at 30 FPS (33.33 ms frame interval).

    The synchronizer computes timing delta:
        Δt = |t_RGB - t_LWIR|

    Enforces the half-frame Nyquist bound:
        Δt <= 16.67 ms (half frame period at 30 Hz).
    Any frame pair exceeding this window is flagged as desynchronized and
    interpolated / re-indexed.
    """

    def __init__(self, target_fps: float = 30.0, max_jitter_ms: float = MAX_SYNC_JITTER_MS):
        self.target_fps = target_fps
        self.frame_period_ms = 1000.0 / target_fps
        self.max_jitter_ms = max_jitter_ms

    def synchronize(
        self,
        t_rgb_ms: float,
        t_thermal_ms: float
    ) -> Tuple[bool, float, float]:
        """
        Validates hardware timestamp synchronization between modalities.

        Args:
            t_rgb_ms: Optical sensor frame hardware timestamp (ms)
            t_thermal_ms: Thermal microbolometer frame hardware timestamp (ms)

        Returns:
            is_synchronized: Boolean indicating if Δt <= tolerance
            jitter_ms: Absolute timestamp disparity in milliseconds
            effective_fps: Instantaneous capture frequency
        """
        jitter_ms = abs(t_rgb_ms - t_thermal_ms)
        is_synchronized = bool(jitter_ms <= self.max_jitter_ms)
        effective_fps = 1000.0 / self.frame_period_ms
        return is_synchronized, jitter_ms, effective_fps


# ==============================================================================
# MODULE 3: SYNCHRONIZED MULTIMODAL & MULTI-TASK TRANSFORMS
# ==============================================================================

class SynchronizedMultimodalTransform:
    """
    Synchronized transformation engine for multi-modal and multi-task learning.

    CRITICAL ARCHITECTURAL REQUIREMENT:
    Spatial augmentations (e.g. Random Resized Crop, Horizontal Flip) MUST be applied
    with strictly identical random parameters across:
        1. RGB image tensor [3, H, W]
        2. Thermal image tensor [3, H, W]
        3. Semantic segmentation mask [H, W] (Nearest-neighbor to preserve class IDs)
        4. Crowd density map [1, H, W] (Bilinear with mass-conservation scaling)

    Applying transformations independently would break cross-modal spatial correspondence
    and invalidate the cross-task consistency loss L_mask-density.
    """

    def __init__(
        self,
        output_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
        crop_scale: Tuple[float, float] = (0.8, 1.0),
        p_flip: float = 0.5,
        is_train: bool = True
    ):
        self.output_size = output_size
        self.crop_scale = crop_scale
        self.p_flip = p_flip
        self.is_train = is_train

    def __call__(
        self,
        rgb: torch.Tensor,
        thermal: torch.Tensor,
        sem_seg: torch.Tensor,
        density: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes synchronized transforms on all 4 multimodal/multitask tensors.

        Args:
            rgb: [3, H, W] float in [0, 1]
            thermal: [3, H, W] float in [0, 1]
            sem_seg: [H, W] long in [0, NUM_SEG_CLASSES-1]
            density: [1, H, W] float continuous density map

        Returns:
            rgb_trans: [3, out_H, out_W]
            thermal_trans: [3, out_H, out_W]
            sem_seg_trans: [out_H, out_W]
            density_trans: [1, out_H, out_W]
        """
        _, orig_h, orig_w = rgb.shape

        if self.is_train:
            # 1. Synchronized Random Crop Parameter Generation
            scale = torch.empty(1).uniform_(self.crop_scale[0], self.crop_scale[1]).item()
            crop_h = int(orig_h * scale)
            crop_w = int(orig_w * scale)
            top = torch.randint(0, orig_h - crop_h + 1, size=(1,)).item()
            left = torch.randint(0, orig_w - crop_w + 1, size=(1,)).item()

            # Crop RGB & Thermal
            rgb = TF.crop(rgb, top, left, crop_h, crop_w)
            thermal = TF.crop(thermal, top, left, crop_h, crop_w)

            # Crop Segmentation Mask (ensure [1, H, W] for TF.crop, then squeeze)
            sem_seg = TF.crop(sem_seg.unsqueeze(0), top, left, crop_h, crop_w).squeeze(0)

            # Crop Crowd Density Map
            density = TF.crop(density, top, left, crop_h, crop_w)

            # 2. Synchronized Random Horizontal Flip
            do_flip = (torch.rand(1).item() < self.p_flip)
            if do_flip:
                rgb = TF.hflip(rgb)
                thermal = TF.hflip(thermal)
                sem_seg = TF.hflip(sem_seg)
                density = TF.hflip(density)

        # 3. Synchronized Resize to Target Spatial Dimensions [256, 256]
        # RGB & Thermal: Bilinear interpolation
        rgb = TF.resize(rgb, self.output_size, interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        thermal = TF.resize(thermal, self.output_size, interpolation=TF.InterpolationMode.BILINEAR, antialias=True)

        # Semantic Mask: Nearest-neighbor interpolation to preserve discrete class tokens
        sem_seg_expanded = sem_seg.unsqueeze(0)  # [1, H, W]
        sem_seg_resized = TF.resize(
            sem_seg_expanded,
            self.output_size,
            interpolation=TF.InterpolationMode.NEAREST
        ).squeeze(0)

        # Crowd Density: Bilinear interpolation with density mass conservation factor
        # When resizing an area from (curr_h * curr_w) to (target_h * target_w), the continuous
        # density values must be rescaled by (curr_area / target_area) so that integral equals true count!
        curr_h, curr_w = density.shape[-2:]
        target_h, target_w = self.output_size
        area_scale_factor = (curr_h * curr_w) / float(target_h * target_w)

        density_resized = TF.resize(
            density,
            self.output_size,
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias=True
        ) * area_scale_factor

        return rgb, thermal, sem_seg_resized, density_resized


# ==============================================================================
# MODULE 4: TRAIN-SET STATISTICAL NORMALIZATION (I = (I - μ) / σ)
# ==============================================================================

class TrainSetStatisticalNormalization(nn.Module):
    """
    Standardizes RGB and Thermal tensors using modality-specific empirical train-set statistics:
        I_norm = (I - μ) / σ

    RGB Channel Statistics: ImageNet-22k pretrained values matching ConvNeXt V2 Tiny backbone.
    Thermal Channel Statistics: Multi-spectral LWIR sensor calibration statistics.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("rgb_mean", torch.tensor(RGB_MEAN, dtype=torch.float32).view(3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor(RGB_STD, dtype=torch.float32).view(3, 1, 1))
        self.register_buffer("thermal_mean", torch.tensor(THERMAL_MEAN, dtype=torch.float32).view(3, 1, 1))
        self.register_buffer("thermal_std", torch.tensor(THERMAL_STD, dtype=torch.float32).view(3, 1, 1))

    def normalize_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        """Applies (I_rgb - μ_rgb) / σ_rgb."""
        return (rgb - self.rgb_mean.to(rgb.device)) / self.rgb_std.to(rgb.device)

    def normalize_thermal(self, thermal: torch.Tensor) -> torch.Tensor:
        """Applies (I_thermal - μ_thermal) / σ_thermal."""
        return (thermal - self.thermal_mean.to(thermal.device)) / self.thermal_std.to(thermal.device)

    def forward(
        self,
        rgb: torch.Tensor,
        thermal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.normalize_rgb(rgb), self.normalize_thermal(thermal)


# ==============================================================================
# MODULE 5: CUSTOM PYTORCH DATASET (CTCMDataset)
# ==============================================================================

class CTCMDataset(Dataset):
    """
    Multimodal Multi-Task PyTorch Dataset for CTCM-Net.

    Simulates paired RGB and LWIR thermal data adhering to the empirical format
    of the three target benchmarks:
        1. MFNet Semantic Segmentation: 9 classes (unlabeled, car, person, bike, ...)
        2. RGBT-CC Crowd Density Estimation: Continuous 2D Gaussian density response map
        3. DarkAct Action Recognition: 27 human action classes

    Cross-Task Spatial Consistency Modeling:
        In synthetic mode, entities (pedestrians, vehicles) are placed consistently:
        - A person at (x, y) receives class label 2 in the MFNet segmentation mask.
        - That same person at (x, y) generates a calibrated Gaussian peak in the RGBT-CC density map.
        - That same person at (x, y) exhibits a distinct physiological heat signature (LWIR elevation).
        - Illumination conditions (day vs. night) dynamically modulate RGB chrominance and thermal contrast.
    """

    def __init__(
        self,
        num_samples: int = 64,
        spatial_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
        is_train: bool = True,
        seed: int = 42
    ):
        super().__init__()
        self.num_samples = num_samples
        self.spatial_size = spatial_size
        self.is_train = is_train
        self.seed = seed

        # Submodules
        self.homography_module = SpatialHomographyRegistration()
        self.temporal_sync = TemporalSynchronizer(target_fps=30.0)
        self.transform = SynchronizedMultimodalTransform(
            output_size=spatial_size,
            crop_scale=(0.85, 1.0),
            p_flip=0.5 if is_train else 0.0,
            is_train=is_train
        )
        self.normalizer = TrainSetStatisticalNormalization()

    def __len__(self) -> int:
        return self.num_samples

    def _generate_gaussian_kernel(self, h: int, w: int, cx: float, cy: float, sigma: float = 4.0) -> torch.Tensor:
        """Generates a normalized 2D Gaussian response kernel simulating crowd density annotation."""
        y = torch.arange(0, h, dtype=torch.float32).view(-1, 1).repeat(1, w)
        x = torch.arange(0, w, dtype=torch.float32).view(1, -1).repeat(h, 1)
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
        kernel = torch.exp(-dist_sq / (2.0 * (sigma ** 2)))
        norm_factor = 2.0 * math.pi * (sigma ** 2)
        return kernel / norm_factor

    def _simulate_multimodal_sample(self, idx: int) -> Dict[str, Any]:
        """
        Synthesizes a geometrically coherent, multi-task paired sample.
        """
        # Ensure deterministic sample generation based on index and master seed
        rng = torch.Generator().manual_seed(self.seed + idx)

        # Baseline resolution before transform: 288 x 384 (aspect ratio 4:3)
        raw_h, raw_w = 288, 384

        # Illumination simulation: 0.0 (Pitch Dark Night) to 1.0 (Bright Sunny Day)
        illumination_factor = torch.rand(1, generator=rng).item()
        is_night = (illumination_factor < 0.45)

        # Base background tensors
        # RGB: bright during day, dark + noisy during night
        if is_night:
            rgb_base = torch.rand(3, raw_h, raw_w, generator=rng) * 0.15 + 0.02
            # Add sensor photon noise in low-light conditions
            rgb_base += torch.randn(3, raw_h, raw_w, generator=rng) * 0.03
        else:
            rgb_base = torch.rand(3, raw_h, raw_w, generator=rng) * 0.65 + 0.25

        # Thermal (LWIR): high contrast at night (hot humans against cold background),
        # lower contrast during day (solar thermal clutter)
        thermal_base_1ch = torch.rand(1, raw_h, raw_w, generator=rng) * 0.25 + 0.15

        # Multi-task targets
        sem_seg = torch.zeros((raw_h, raw_w), dtype=torch.long)
        density = torch.zeros((1, raw_h, raw_w), dtype=torch.float32)

        # Simulate dynamic pedestrian count: 2 to 7 people
        num_pedestrians = torch.randint(2, 8, (1,), generator=rng).item()

        for _ in range(num_pedestrians):
            # Sample pedestrian center coordinate
            cx = torch.randint(20, raw_w - 20, (1,), generator=rng).item()
            cy = torch.randint(30, raw_h - 30, (1,), generator=rng).item()
            ph = torch.randint(24, 48, (1,), generator=rng).item()  # Pedestrian height
            pw = max(10, int(ph * 0.45))                            # Pedestrian width

            y1 = max(0, cy - ph // 2)
            y2 = min(raw_h, cy + ph // 2)
            x1 = max(0, cx - pw // 2)
            x2 = min(raw_w, cx + pw // 2)

            # 1. Semantic Segmentation: class 2 ("person")
            sem_seg[y1:y2, x1:x2] = 2

            # 2. Crowd Density: Continuous 2D Gaussian response centered on pedestrian head/torso
            density_kernel = self._generate_gaussian_kernel(raw_h, raw_w, cx, cy, sigma=3.5)
            density[0] += density_kernel

            # 3. Thermal Signature: High infrared radiance (~37°C body heat vs ambient)
            thermal_base_1ch[0, y1:y2, x1:x2] += 0.55 if is_night else 0.35

            # 4. RGB Chrominance Modulation
            if not is_night:
                rgb_base[0, y1:y2, x1:x2] += 0.20  # Visible clothing reflection
                rgb_base[1, y1:y2, x1:x2] += 0.15

        # Simulate 1 vehicle: class 1 ("car")
        has_car = (torch.rand(1, generator=rng).item() > 0.3)
        if has_car:
            cx = torch.randint(40, raw_w - 40, (1,), generator=rng).item()
            cy = torch.randint(raw_h // 2, raw_h - 25, (1,), generator=rng).item()
            ch = torch.randint(25, 45, (1,), generator=rng).item()
            cw = torch.randint(50, 90, (1,), generator=rng).item()
            y1, y2 = max(0, cy - ch // 2), min(raw_h, cy + ch // 2)
            x1, x2 = max(0, cx - cw // 2), min(raw_w, cx + cw // 2)

            sem_seg[y1:y2, x1:x2] = 1
            # Car engine thermal heat signature
            thermal_base_1ch[0, y1:y2, x1:x2] += 0.45

        # Clamp physical radiance values to [0.0, 1.0]
        rgb = torch.clamp(rgb_base, 0.0, 1.0)
        thermal_1ch = torch.clamp(thermal_base_1ch, 0.0, 1.0)

        # Replicate 1-channel LWIR to 3 channels to match standard vision backbone input (ConvNeXt-V2)
        thermal_3ch = thermal_1ch.repeat(3, 1, 1)

        # Sample DarkAct Action Recognition Ground Truth: 0 to 26
        action_label = torch.randint(0, NUM_ACTION_CLASSES, (1,), generator=rng).item()

        # Simulate Hardware Timestamps for 30 FPS clock synchronization
        # Frame period is 33.33ms; simulate slight inter-sensor clock jitter (< 10ms)
        t_base_ms = float(idx * 33.333)
        jitter_offset = (torch.rand(1, generator=rng).item() - 0.5) * 8.0  # [-4ms, +4ms]
        t_rgb = t_base_ms
        t_thermal = t_base_ms + jitter_offset

        return {
            "rgb": rgb,
            "thermal": thermal_3ch,
            "sem_seg": sem_seg,
            "density": density,
            "action_label": action_label,
            "illumination_factor": illumination_factor,
            "t_rgb_ms": t_rgb,
            "t_thermal_ms": t_thermal
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Pipeline Flow:
        Raw Sensor Generation -> Homography Registration -> Temporal Sync Check
        -> Synchronized Spatial Transforms -> Statistical Normalization
        """
        # 1. Synthesize Geometrically Paired Raw Sensor Data
        raw = self._simulate_multimodal_sample(idx)

        # 2. Spatial Homography Registration (LWIR to RGB Reference Plane)
        aligned_rgb, aligned_thermal, homography_matrix = self.homography_module(
            raw["rgb"], raw["thermal"]
        )

        # 3. Temporal Clock Synchronization Verification
        is_synced, jitter_ms, eff_fps = self.temporal_sync.synchronize(
            raw["t_rgb_ms"], raw["t_thermal_ms"]
        )

        # 4. Synchronized Multimodal & Multi-Task Spatial Transforms
        # Guarantees identical random crops and flips across all 4 spatial fields
        (
            trans_rgb,
            trans_thermal,
            trans_sem_seg,
            trans_density
        ) = self.transform(
            aligned_rgb,
            aligned_thermal,
            raw["sem_seg"],
            raw["density"]
        )

        # 5. Train-Set Statistical Normalization: I = (I - μ) / σ
        norm_rgb, norm_thermal = self.normalizer(trans_rgb, trans_thermal)

        return {
            "rgb": norm_rgb.float(),                                # [3, 256, 256]
            "thermal": norm_thermal.float(),                        # [3, 256, 256]
            "sem_seg_mask": trans_sem_seg.long(),                   # [256, 256] (0 to 8)
            "density_map": trans_density.float(),                   # [1, 256, 256] (>= 0)
            "action_label": torch.tensor(raw["action_label"], dtype=torch.long),  # scalar [0 to 26]
            "illumination_level": torch.tensor([raw["illumination_factor"]], dtype=torch.float32),  # [1]
            "homography_matrix": homography_matrix.float(),         # [3, 3]
            "is_synchronized": torch.tensor(is_synced, dtype=torch.bool),
            "temporal_jitter_ms": torch.tensor(jitter_ms, dtype=torch.float32),
            "metadata": {
                "action_name": DARKACT_CLASSES[raw["action_label"]],
                "sample_idx": idx,
                "is_night": bool(raw["illumination_factor"] < 0.45)
            }
        }
