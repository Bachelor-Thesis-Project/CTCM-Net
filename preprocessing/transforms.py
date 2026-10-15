"""
CTCM-Net: Preprocessing Step 3 - Aligned Multimodal Data Augmentation Engine
=============================================================================
Module: preprocessing/transforms.py
Hardware Target: NVIDIA RTX A6000

Step 3 Requirements:
- Symmetric spatial transformations (RandomHorizontalFlip, RandomCrop).
- MUST apply the exact same geometric transformation to the RGB tensors,
  Thermal tensors, and the target masks/heatmaps simultaneously to preserve
  paired multimodal and cross-task correspondence.
- Density Map: Bilinear interpolation with mass-conservation scaling factor:
    area_scale = (H_in * W_in) / (H_out * W_out)
- Segmentation Mask: Nearest-neighbor interpolation to preserve discrete class tokens.
- Video Sequences: Applies identical spatial crop and flip parameters across
  all 16 frames of both RGB and Thermal video streams.
"""

from typing import Optional, Tuple, Union
import torch
import torchvision.transforms.functional as TF


TARGET_RESOLUTION: Tuple[int, int] = (256, 256)


class AlignedMultimodalTransform:
    """
    Synchronized spatial transformation engine for multimodal paired inputs.

    Applies deterministic, coupled spatial augmentations with strictly identical
    parameters across:
        1. RGB image/sequence [3, H, W] or [T, 3, H, W]
        2. Thermal image/sequence [3, H, W] or [T, 3, H, W]
        3. Target (optional):
           - Semantic segmentation mask [H, W] (Nearest-neighbor)
           - Continuous crowd density map [1, H, W] (Mass-conserved bilinear)
    """

    def __init__(
        self,
        output_size: Tuple[int, int] = TARGET_RESOLUTION,
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
        target: Optional[torch.Tensor] = None,
        is_density: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Executes synchronized transforms on RGB, Thermal, and optional Target.

        Args:
            rgb: [3, H, W] or [T, 3, H, W] float in [0, 1]
            thermal: [3, H, W] or [T, 3, H, W] float in [0, 1]
            target: Optional target tensor:
                    - [H, W] LongTensor for segmentation mask
                    - [1, H, W] FloatTensor for crowd density map
            is_density: True if target is a continuous density map

        Returns:
            rgb_trans: [3, 256, 256] or [T, 3, 256, 256]
            thermal_trans: [3, 256, 256] or [T, 3, 256, 256]
            target_trans: [256, 256] or [1, 256, 256] (if target was provided)
        """
        is_video = (rgb.dim() == 4)  # [T, C, H, W]
        orig_h, orig_w = rgb.shape[-2:]

        # 1. Determine Crop Box (Coupled Parameter Generation)
        if self.is_train and self.crop_scale[0] < 1.0:
            scale = torch.empty(1).uniform_(self.crop_scale[0], self.crop_scale[1]).item()
            crop_h = max(32, int(orig_h * scale))
            crop_w = max(32, int(orig_w * scale))
            top = int(torch.randint(0, max(1, orig_h - crop_h + 1), size=(1,)).item())
            left = int(torch.randint(0, max(1, orig_w - crop_w + 1), size=(1,)).item())
        else:
            top, left, crop_h, crop_w = 0, 0, orig_h, orig_w

        # 2. Determine Flip Decision
        do_flip = bool(self.is_train and (torch.rand(1).item() < self.p_flip))

        # 3. Apply Transformations to RGB & Thermal
        if is_video:
            # Video: [T, C, H, W] - loop through frames or apply batch crop
            T, C, _, _ = rgb.shape
            rgb_cropped = TF.crop(rgb, top, left, crop_h, crop_w)
            th_cropped = TF.crop(thermal, top, left, crop_h, crop_w)

            if do_flip:
                rgb_cropped = TF.hflip(rgb_cropped)
                th_cropped = TF.hflip(th_cropped)

            # Resize to output size [256, 256]
            rgb_trans = TF.resize(
                rgb_cropped,
                list(self.output_size),
                interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True
            )
            thermal_trans = TF.resize(
                th_cropped,
                list(self.output_size),
                interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True
            )
        else:
            # Single Image Pair: [C, H, W]
            rgb_cropped = TF.crop(rgb, top, left, crop_h, crop_w)
            th_cropped = TF.crop(thermal, top, left, crop_h, crop_w)

            if do_flip:
                rgb_cropped = TF.hflip(rgb_cropped)
                th_cropped = TF.hflip(th_cropped)

            rgb_trans = TF.resize(
                rgb_cropped,
                list(self.output_size),
                interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True
            )
            thermal_trans = TF.resize(
                th_cropped,
                list(self.output_size),
                interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True
            )

        # 4. Apply Coupled Transformations to Target (if provided)
        target_trans = None
        if target is not None:
            if is_density:
                # Target is continuous density map [1, H, W]
                # Step A: Crop density map
                density_crop = TF.crop(target, top, left, crop_h, crop_w)
                # Step B: Flip if required
                if do_flip:
                    density_crop = TF.hflip(density_crop)
                # Step C: Resize with mass-conservation scaling factor
                # When resizing an area from (crop_h * crop_w) to (target_h * target_w),
                # the continuous density values must be multiplied by (crop_area / target_area)
                # to guarantee the integral sum equals the true crowd count!
                target_h, target_w = self.output_size
                mass_factor = float(crop_h * crop_w) / float(target_h * target_w)

                target_trans = TF.resize(
                    density_crop,
                    list(self.output_size),
                    interpolation=TF.InterpolationMode.BILINEAR,
                    antialias=True
                ) * mass_factor
            else:
                # Target is discrete segmentation mask [H, W] or [1, H, W]
                is_2d = (target.dim() == 2)
                mask_in = target.unsqueeze(0) if is_2d else target

                mask_crop = TF.crop(mask_in, top, left, crop_h, crop_w)
                if do_flip:
                    mask_crop = TF.hflip(mask_crop)

                # Nearest neighbor interpolation to strictly preserve discrete class tokens
                mask_resized = TF.resize(
                    mask_crop,
                    list(self.output_size),
                    interpolation=TF.InterpolationMode.NEAREST
                )

                target_trans = mask_resized.squeeze(0) if is_2d else mask_resized

        return rgb_trans, thermal_trans, target_trans
