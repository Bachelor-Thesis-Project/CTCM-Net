"""
CTCM-Net: Cross-Task Consistency Multi-Modal Network
=====================================================
Phase 4: Multi-Task Optimization, Homoscedastic Uncertainty Balancing & Evaluation Suite

This module provides the production-grade optimization and evaluation infrastructure:
1. HomoscedasticUncertaintyBalancer: Custom PyTorch module based on Kendall et al. (CVPR 2018)
   that dynamically weights competing multi-task losses using task-dependent homoscedastic
   uncertainty parameters (σ_1, σ_2, σ_3, σ_4).
2. MultiTaskLossEngine: Supervised loss calculator computing:
   - Semantic Segmentation: Cross-Entropy Loss (MFNet 9 classes)
   - Crowd Density Estimation: Mean Squared Error (RGBT-CC continuous map)
   - Action Recognition: Cross-Entropy Loss (DarkAct 27 classes)
   - Cross-Task Consistency: Bidirectional L_mask-density
3. EvaluationMetricsEngine: Computes standard vision benchmarks:
   - Segmentation: mIoU, Pixel Accuracy
   - Crowd Counting: GAME(0) (MAE), RMSE
   - Action Recognition: Top-1 Accuracy, Top-5 Accuracy
4. train_epoch: Simulates an optimization step with differential learning rates,
   gradient backpropagation, and metric tracking.

Authors: CTCM-Net Research Team
Target Hardware: NVIDIA RTX A6000 (48GB VRAM)
"""

from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import CTCM_Net


# ==============================================================================
# MODULE 1: HOMOSCEDASTIC UNCERTAINTY BALANCER (Kendall et al., CVPR 2018)
# ==============================================================================

class HomoscedasticUncertaintyBalancer(nn.Module):
    """
    Multi-Task Homoscedastic Uncertainty Loss Balancer.

    Based on Kendall et al. (CVPR 2018), "Multi-Task Learning Using Uncertainty to
    Weigh Losses for Scene Geometry and Semantics".

    Dynamically weights competing multi-task losses without manual hyperparameter
    tuning or negative transfer by modeling task-dependent homoscedastic uncertainty.

    Mathematical Formulation:
        L_total = Σ [ (1 / (2 * σ_i^2)) * L_i + log(1 + σ_i^2) ]

    Where:
        σ_1: Semantic Segmentation uncertainty (MFNet)
        σ_2: Crowd Density Estimation uncertainty (RGBT-CC)
        σ_3: Action Recognition uncertainty (DarkAct)
        σ_4: Cross-Task Consistency uncertainty (L_mask-density)

    Properties:
        - When a task has high loss or noisy gradients, σ_i naturally increases,
          which down-weights that task's gradient (1 / (2 * σ_i^2)) to prevent dominating the network.
        - The regularization term log(1 + σ_i^2) acts as a barrier preventing σ_i from diverging to infinity.
    """

    def __init__(
        self,
        num_tasks: int = 4,
        init_sigma: float = 1.0,
        task_names: Optional[List[str]] = None
    ):
        super().__init__()
        self.num_tasks = num_tasks
        if task_names is None:
            self.task_names = [
                "Semantic Segmentation",
                "Crowd Density Estimation",
                "Action Recognition",
                "Cross-Task Consistency"
            ]
        else:
            self.task_names = task_names

        # 4 learnable homoscedastic uncertainty parameters (σ_1, σ_2, σ_3, σ_4)
        # Initialized to 1.0, giving an initial loss weight of 1 / (2 * 1^2) = 0.5
        self.sigmas = nn.Parameter(torch.full((num_tasks,), init_sigma, dtype=torch.float32))

    def forward(
        self,
        task_losses: List[torch.Tensor],
        eps: float = 1e-6
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            task_losses: List of 4 scalar loss tensors [L_seg, L_density, L_action, L_consistency]

        Returns:
            loss_total: Balanced scalar multi-task objective
            metadata: Detailed loss components, effective weights, and uncertainties
        """
        assert len(task_losses) == self.num_tasks, (
            f"Expected {self.num_tasks} task losses, got {len(task_losses)}"
        )

        device = task_losses[0].device
        total_loss = torch.tensor(0.0, device=device)
        weights_record = []
        reg_record = []
        effective_weights = []

        for i, l_i in enumerate(task_losses):
            sigma_i = self.sigmas[i]
            sigma_sq = (sigma_i ** 2) + eps

            # Inverse variance weighting: 1 / (2 * σ_i^2)
            w_i = 1.0 / (2.0 * sigma_sq)

            # Log-variance regularization penalty: log(1 + σ_i^2)
            reg_i = torch.log(1.0 + sigma_sq)

            # Weighted task loss contribution
            task_contribution = (w_i * l_i) + reg_i
            total_loss = total_loss + task_contribution

            weights_record.append(w_i.item())
            reg_record.append(reg_i.item())
            effective_weights.append({
                "task_name": self.task_names[i],
                "raw_loss": l_i.item(),
                "sigma": sigma_i.item(),
                "sigma_sq": sigma_sq.item(),
                "weight": w_i.item(),
                "reg_penalty": reg_i.item(),
                "weighted_contribution": task_contribution.item()
            })

        metadata = {
            "loss_total": total_loss.item(),
            "weights": weights_record,
            "sigmas": self.sigmas.detach().cpu().numpy().tolist(),
            "effective_weights": effective_weights
        }

        return total_loss, metadata


# ==============================================================================
# MODULE 2: MULTI-TASK LOSS ENGINE
# ==============================================================================

class MultiTaskLossEngine(nn.Module):
    """
    Computes standard task-specific supervised losses:
    1. Semantic Segmentation: Cross-Entropy Loss over 9 classes
    2. Crowd Density Estimation: Mean Squared Error (continuous spatial response)
    3. Action Recognition: Cross-Entropy Loss over 27 classes
    4. Cross-Task Consistency: Bidirectional L_mask-density (from Consistency Engine)
    """

    def __init__(self, ignore_index: int = 0):
        super().__init__()
        # Ignore unlabelled pixels (class 0) if desired, or include all 9 classes
        self.seg_criterion = nn.CrossEntropyLoss()
        self.density_criterion = nn.MSELoss()
        self.action_criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        predictions: Dict[str, Any],
        targets: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: Output dictionary from CTCM_Net forward pass
            targets: Ground truth batch dictionary from DataLoader

        Returns:
            Dictionary containing individual raw task losses and decomposed components
        """
        seg_logits = predictions["seg_logits"]          # [B, 9, 256, 256]
        density_pred = predictions["density_map"]        # [B, 1, 256, 256]
        action_logits = predictions["action_logits"]      # [B, 27]

        seg_gt = targets["sem_seg_mask"]                 # [B, 256, 256]
        density_gt = targets["density_map"]              # [B, 1, 256, 256]
        action_gt = targets["action_label"]              # [B]

        # 1. Semantic Segmentation Cross-Entropy Loss
        l_seg = self.seg_criterion(seg_logits, seg_gt)

        # 2. Crowd Density Estimation MSE Loss
        l_density = self.density_criterion(density_pred, density_gt)

        # 3. Action Recognition Cross-Entropy Loss
        l_action = self.action_criterion(action_logits, action_gt)

        # 4. Cross-Task Consistency Loss (L_mask-density)
        l_consistency = predictions["consistency_outputs"]["loss_mask_density"]

        # Phase 2 Decomposition Regularization Loss (L_sim & L_diff)
        l_decomp = predictions["loss_decomp"]["loss_decomp_total"]

        return {
            "l_seg": l_seg,
            "l_density": l_density,
            "l_action": l_action,
            "l_consistency": l_consistency,
            "l_decomp": l_decomp
        }


# ==============================================================================
# MODULE 3: EVALUATION METRICS ENGINE
# ==============================================================================

class EvaluationMetricsEngine:
    """
    Computes academic-standard evaluation metrics across all 3 vision tasks:
    1. Semantic Segmentation:
       - Mean Intersection over Union (mIoU) across 9 classes
       - Pixel Accuracy
    2. Crowd Density Estimation:
       - GAME(0) / Mean Absolute Error (MAE) of total count: |N_pred - N_gt|
       - Root Mean Squared Error (RMSE) of total count: sqrt( (N_pred - N_gt)^2 )
    3. Action Recognition:
       - Top-1 Accuracy (%)
       - Top-5 Accuracy (%)
    """

    def __init__(self, num_seg_classes: int = 9, num_action_classes: int = 27):
        self.num_seg_classes = num_seg_classes
        self.num_action_classes = num_action_classes

    def compute_batch_metrics(
        self,
        seg_logits: torch.Tensor,
        seg_gt: torch.Tensor,
        density_pred: torch.Tensor,
        density_gt: torch.Tensor,
        action_logits: torch.Tensor,
        action_gt: torch.Tensor,
        eps: float = 1e-7
    ) -> Dict[str, float]:
        with torch.no_grad():
            # 1. Semantic Segmentation Metrics
            pred_mask = torch.argmax(seg_logits, dim=1)  # [B, 256, 256]
            correct_pixels = (pred_mask == seg_gt).sum().item()
            total_pixels = seg_gt.numel()
            pixel_acc = (correct_pixels / total_pixels) * 100.0

            ious = []
            for c in range(self.num_seg_classes):
                tp = ((pred_mask == c) & (seg_gt == c)).sum().item()
                fp = ((pred_mask == c) & (seg_gt != c)).sum().item()
                fn = ((pred_mask != c) & (seg_gt == c)).sum().item()
                denom = tp + fp + fn
                if denom > 0:
                    ious.append(tp / (denom + eps))
            miou = (sum(ious) / max(1, len(ious))) * 100.0

            # 2. Crowd Density Metrics: count = sum(density_map)
            pred_counts = density_pred.sum(dim=(-2, -1)).squeeze(-1)  # [B]
            gt_counts = density_gt.sum(dim=(-2, -1)).squeeze(-1)      # [B]

            count_errors = (pred_counts - gt_counts).abs()
            game0 = count_errors.mean().item()                        # GAME(0) is MAE
            rmse = torch.sqrt((count_errors ** 2).mean()).item()

            # 3. Action Recognition Metrics
            pred_top1 = torch.argmax(action_logits, dim=1)            # [B]
            top1_acc = (pred_top1 == action_gt).float().mean().item() * 100.0

            k = min(5, self.num_action_classes)
            top5_preds = torch.topk(action_logits, k=k, dim=1).indices # [B, 5]
            top5_acc = torch.any(top5_preds == action_gt.unsqueeze(1), dim=1).float().mean().item() * 100.0

        return {
            "miou": miou,
            "pixel_acc": pixel_acc,
            "game0": game0,
            "rmse": rmse,
            "top1_acc": top1_acc,
            "top5_acc": top5_acc,
            "mean_pred_count": pred_counts.mean().item(),
            "mean_gt_count": gt_counts.mean().item()
        }


# ==============================================================================
# MODULE 4: TRAINING ITERATION ENGINE (train_epoch)
# ==============================================================================

def train_epoch(
    model: CTCM_Net,
    balancer: HomoscedasticUncertaintyBalancer,
    loss_engine: MultiTaskLossEngine,
    metrics_engine: EvaluationMetricsEngine,
    batch: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device
) -> Dict[str, Any]:
    """
    Executes a complete training step:
    1. Records initial homoscedastic uncertainty weights (σ_initial)
    2. Forward pass through CTCM_Net
    3. Computes individual task losses & Phase 2 decomposition losses
    4. Evaluates Homoscedastic Uncertainty Balancing:
       L_total = Σ [ (1 / (2 * σ_i^2)) * L_i + log(1 + σ_i^2) ] + L_decomp
    5. Backpropagation (backward())
    6. Verifies non-zero gradient flow across all modules
    7. Optimizer step (optimizer.step()) & LR scheduler step
    8. Records updated homoscedastic uncertainty weights (σ_updated)
    9. Computes evaluation metrics
    """
    model.train()
    balancer.train()

    # Move batch tensors to device
    rgb = batch["rgb"].to(device)
    thermal = batch["thermal"].to(device)
    illumination = batch["illumination_level"].to(device)
    sem_seg_gt = batch["sem_seg_mask"].to(device)
    density_gt = batch["density_map"].to(device)
    action_gt = batch["action_label"].to(device)

    targets = {
        "sem_seg_mask": sem_seg_gt,
        "density_map": density_gt,
        "action_label": action_gt
    }

    # 1. Capture initial sigma values
    sigmas_initial = balancer.sigmas.detach().clone().cpu().numpy().tolist()

    # 2. Forward pass through unified CTCM_Net
    predictions = model(rgb, thermal, illumination)

    # 3. Compute raw multi-task losses
    raw_losses = loss_engine(predictions, targets)
    l_seg = raw_losses["l_seg"]
    l_density = raw_losses["l_density"]
    l_action = raw_losses["l_action"]
    l_consistency = raw_losses["l_consistency"]
    l_decomp = raw_losses["l_decomp"]

    # 4. Homoscedastic Uncertainty Balancing across 4 competing losses
    task_loss_list = [l_seg, l_density, l_action, l_consistency]
    loss_balanced, balancer_meta = balancer(task_loss_list)

    # Total loss includes balanced task losses plus Phase 2 decomposition regularization
    loss_total = loss_balanced + (0.1 * l_decomp)

    # 5. Backward Pass
    optimizer.zero_grad()
    loss_total.backward()

    # 6. Gradient Flow Audit across all modular sub-networks
    def get_grad_norm(module: nn.Module) -> float:
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        if not grads:
            return 0.0
        return torch.norm(torch.stack([torch.norm(g) for g in grads])).item()

    grad_norms = {
        "backbone_rgb": get_grad_norm(model.phase2_engine.backbone.rgb_encoder),
        "backbone_thermal": get_grad_norm(model.phase2_engine.backbone.thermal_encoder),
        "decomposition_fusion": get_grad_norm(model.phase2_engine.decomp_stages),
        "seg_head": get_grad_norm(model.seg_head),
        "density_head": get_grad_norm(model.density_head),
        "action_head": get_grad_norm(model.action_head),
        "balancer": get_grad_norm(balancer)
    }

    # Gradient clipping for numerical stability
    torch.nn.utils.clip_grad_norm_(
        list(model.parameters()) + list(balancer.parameters()),
        max_norm=5.0
    )

    # 7. Optimizer & Scheduler Updates
    optimizer.step()
    scheduler.step()

    # 8. Capture updated sigma values and deltas
    sigmas_updated = balancer.sigmas.detach().clone().cpu().numpy().tolist()
    sigma_deltas = [u - i for i, u in zip(sigmas_initial, sigmas_updated)]

    # 9. Compute Evaluation Metrics
    batch_metrics = metrics_engine.compute_batch_metrics(
        predictions["seg_logits"],
        sem_seg_gt,
        predictions["density_map"],
        density_gt,
        predictions["action_logits"],
        action_gt
    )

    return {
        "loss_total": loss_total.item(),
        "loss_balanced": loss_balanced.item(),
        "raw_losses": {
            "l_seg": l_seg.item(),
            "l_density": l_density.item(),
            "l_action": l_action.item(),
            "l_consistency": l_consistency.item(),
            "l_decomp": l_decomp.item()
        },
        "sigmas_initial": sigmas_initial,
        "sigmas_updated": sigmas_updated,
        "sigma_deltas": sigma_deltas,
        "balancer_meta": balancer_meta,
        "grad_norms": grad_norms,
        "batch_metrics": batch_metrics,
        "current_lr_backbone": optimizer.param_groups[0]["lr"],
        "current_lr_heads": optimizer.param_groups[1]["lr"]
    }
