"""
Loss Functions for ColonMamba Training

Implements:
- Focal + IoU combined loss (ColonFormer style) - DEFAULT
- BCE + Dice combined loss (alternative)
- Deep Supervision loss with multi-scale outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    
    Dice = 2*|X ∩ Y| / (|X| + |Y|)
    Dice Loss = 1 - Dice
    """
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): Predictions (logits), shape [B, C, H, W]
            target (Tensor): Ground truth, shape [B, C, H, W]
        
        Returns:
            Tensor: Dice loss (scalar)
        """
        # Apply sigmoid to get probabilities
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        
        # Calculate intersection and union
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        
        # FIX: Proper empty case handling
        dice = torch.zeros_like(intersection, dtype=torch.float32)
        valid_mask = (union > 0)
        
        if valid_mask.any():
            dice[valid_mask] = (2. * intersection[valid_mask]) / union[valid_mask]
        
        # Dice loss
        dice_loss = 1 - dice.mean()
        
        return dice_loss


class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice Loss.
    
    This combination works well for segmentation:
    - BCE: Pixel-wise classification
    - Dice: Global overlap optimization
    
    Args:
        bce_weight (float): Weight for BCE loss (default: 0.5)
        dice_weight (float): Weight for Dice loss (default: 0.5)
        smooth (float): Smoothing factor for Dice (default: 1.0)
    """
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
    
    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): Predictions (logits), shape [B, 1, H, W]
            target (Tensor): Ground truth, shape [B, 1, H, W]
        
        Returns:
            Tensor: Combined loss (scalar)
        """
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        return total_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance and hard examples.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha (float): Weighting factor (default: 0.25)
        gamma (float): Focusing parameter (default: 2.0)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): Predictions (logits), shape [B, 1, H, W]
            target (Tensor): Ground truth, shape [B, 1, H, W]
        
        Returns:
            Tensor: Focal loss (scalar)
        """
        # Get probabilities
        prob = torch.sigmoid(pred)
        
        # Calculate p_t
        p_t = prob * target + (1 - prob) * (1 - target)
        
        # Calculate focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Apply class-conditional alpha (FIXED)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha_t * focal_weight * bce_loss
        
        return focal_loss.mean()


class IoULoss(nn.Module):
    """
    IoU Loss (Jaccard Loss) for segmentation.
    
    IoU = |A ∩ B| / |A ∪ B| = |A ∩ B| / (|A| + |B| - |A ∩ B|)
    IoU Loss = 1 - IoU
    
    Stricter than Dice Loss - better for small objects like polyps.
    
    Args:
        smooth (float): Smoothing factor to avoid division by zero (default: 1.0)
    """
    
    def __init__(self, smooth=1.0):
        super(IoULoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): Predictions (logits), shape [B, C, H, W]
            target (Tensor): Ground truth, shape [B, C, H, W]
        
        Returns:
            Tensor: IoU loss (scalar)
        """
        # Apply sigmoid to get probabilities
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        
        # Calculate intersection and union
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1) - intersection
        
        # FIX: Properly handle empty cases
        # Only calculate IoU where union > 0
        iou = torch.zeros_like(intersection, dtype=torch.float32)
        valid_mask = (union > 0)
        
        if valid_mask.any():
            iou[valid_mask] = intersection[valid_mask] / union[valid_mask]
        
        # Empty predictions should be penalized (iou=0)
        # This prevents model from learning to output all zeros
        
        # IoU loss
        iou_loss = 1 - iou.mean()
        
        return iou_loss


class EdgeAwareWeight(nn.Module):
    """
    Compute edge-aware weight map emphasizing boundaries.
    
    This matches ColonFormer's edge-weighting mechanism in structure_loss:
    weit = 1 + edge_boost * |avg_pool(mask) - mask|
    
    Boundaries have high gradient → high weight (up to 1 + edge_boost)
    Uniform regions have low gradient → low weight (≈ 1)
    
    Args:
        kernel_size (int): Pooling kernel size for gradient computation (default: 31)
        edge_boost (float): Boost factor for edges (default: 5.0)
    """
    
    def __init__(self, kernel_size=31, edge_boost=5.0):
        super(EdgeAwareWeight, self).__init__()
        self.kernel_size = kernel_size
        self.edge_boost = edge_boost
        self.padding = kernel_size // 2
    
    def forward(self, mask):
        """
        Compute edge-aware weight map.
        
        Args:
            mask (Tensor): Ground truth mask [B, 1, H, W] in range [0, 1]
        
        Returns:
            Tensor: Edge-aware weight map [B, 1, H, W] in range [1, 1+edge_boost]
        """
        # Smooth mask via average pooling
        mask_smooth = F.avg_pool2d(
            mask.float(), 
            kernel_size=self.kernel_size, 
            stride=1, 
            padding=self.padding
        )
        
        # Compute gradient magnitude (edge detector)
        # High values at boundaries, low values in uniform regions
        edge_map = torch.abs(mask - mask_smooth)
        
        # Weight = 1 + edge_boost * edge_map
        # Boundaries get weight up to 1 + edge_boost (e.g., 6.0)
        # Background/foreground centers get weight ≈ 1.0
        weight = 1.0 + self.edge_boost * edge_map
        
        return weight


class FocalIoULoss(nn.Module):
    """
    Combined Focal + IoU Loss (ColonFormer style).
    
    This combination works well for polyp segmentation:
    - Focal: Handles class imbalance, focuses on hard examples (boundaries)
    - IoU: Region-based optimization, stricter than Dice
    
    **NEW**: Edge-aware weighting to emphasize boundary regions
    
    Args:
        focal_alpha (float): Alpha for Focal loss (default: 0.25)
        focal_gamma (float): Gamma for Focal loss (default: 2.0)
        focal_weight (float): Weight for Focal loss (default: 0.5)
        iou_weight (float): Weight for IoU loss (default: 0.5)
        iou_smooth (float): Smoothing factor for IoU (default: 1.0)
        use_edge_weight (bool): Use edge-aware weighting (default: True)
        edge_kernel_size (int): Kernel size for edge detection (default: 31)
        edge_boost (float): Edge weight boost factor (default: 5.0)
    """
    
    def __init__(
        self, 
        focal_alpha=0.25,
        focal_gamma=2.0,
        focal_weight=0.5,
        iou_weight=0.5,
        iou_smooth=1.0,
        use_edge_weight=True,
        edge_kernel_size=31,
        edge_boost=5.0,
    ):
        super(FocalIoULoss, self).__init__()
        self.focal_weight = focal_weight
        self.iou_weight = iou_weight
        self.use_edge_weight = use_edge_weight
        
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.iou = IoULoss(smooth=iou_smooth)
        
        if use_edge_weight:
            self.edge_weight_module = EdgeAwareWeight(edge_kernel_size, edge_boost)
    
    def forward(self, pred, target):
        """
        Compute combined focal + IoU loss with optional edge-aware weighting.
        
        Args:
            pred (Tensor): Predictions (logits), shape [B, 1, H, W]
            target (Tensor): Ground truth, shape [B, 1, H, W]
        
        Returns:
            Tensor: Combined loss (scalar)
        """
        if self.use_edge_weight:
            # ===== Edge-aware Weighted Loss (ColonFormer style) =====
            # Compute edge-aware weight map
            weit = self.edge_weight_module(target)  # [B, 1, H, W]
            
            # Weighted Focal Loss
            focal_loss = self.focal(pred, target)  # [B, 1, H, W] (per-pixel)
            focal_loss = (focal_loss * weit).sum() / weit.sum()
            
            # Weighted IoU Loss - FIXED to handle empty cases properly
            pred_sigmoid = torch.sigmoid(pred)
            inter = ((pred_sigmoid * target) * weit).sum()
            union = ((pred_sigmoid + target) * weit).sum()
            
            # FIX: Proper empty case handling
            union_minus_inter = union - inter
            if union_minus_inter > 0:
                iou = inter / union_minus_inter
            else:
                iou = torch.tensor(0.0, device=pred.device)
            
            iou_loss = 1 - iou
            
        else:
            # ===== Standard Loss (no edge weighting) =====
            focal_loss = self.focal(pred, target).mean()
            iou_loss = self.iou(pred, target)
        
        # Combined loss
        total_loss = self.focal_weight * focal_loss + self.iou_weight * iou_loss
        
        return total_loss


class DeepSupervisionLoss(nn.Module):
    """
    Deep Supervision Loss for multi-scale outputs.
    
    Combines losses from main prediction and auxiliary predictions at different scales.
    
    Loss = w_main * L(main, GT) + w_aux1 * L(aux1, GT) + w_aux2 * L(aux2, GT) + ...
    
    Args:
        base_loss (nn.Module): Base loss function (default: FocalIoULoss - ColonFormer style)
        aux_weights (list): Weights for auxiliary outputs (default: [0.4, 0.4, 0.4])
        main_weight (float): Weight for main output (default: 1.0)
    """
    
    def __init__(self, base_loss=None, aux_weights=[0.4, 0.4, 0.4], main_weight=1.0):
        super(DeepSupervisionLoss, self).__init__()
        
        if base_loss is None:
            # Default: ColonFormer-style Focal + IoU
            base_loss = FocalIoULoss()
        
        self.base_loss = base_loss
        self.aux_weights = aux_weights
        self.main_weight = main_weight
   
    def forward(self, outputs, target):
        """
        Args:
            outputs (dict): Model outputs
                'main': [B, 1, H, W]
                'aux_1_8': [B, 1, H/8, W/8]
                'aux_1_16': [B, 1, H/16, W/16]
                'aux_1_32': [B, 1, H/32, W/32]
            target (Tensor): Ground truth mask, shape [B, 1, H, W]
        
        Returns:
            dict: Loss components
                'total': Total loss
                'main': Main branch loss
                'aux_1_8': Auxiliary loss at 1/8 scale
                ...
        """
        loss_dict = {}
        
        # Main loss
        main_loss = self.base_loss(outputs['main'], target)
        loss_dict['main'] = main_loss
        total_loss = self.main_weight * main_loss
        
        # Auxiliary losses (need to downsample target)
        aux_keys = ['aux_1_8', 'aux_1_16', 'aux_1_32']
        
        for i, key in enumerate(aux_keys):
            if key in outputs:
                # Downsample target to match auxiliary output size
                h, w = outputs[key].shape[2:]
                target_down = F.interpolate(target, size=(h, w), mode='nearest')
                
                # Calculate auxiliary loss
                aux_loss = self.base_loss(outputs[key], target_down)
                loss_dict[key] = aux_loss
                
                # Add to total loss with weight
                weight = self.aux_weights[i] if i < len(self.aux_weights) else 0.4
                total_loss += weight * aux_loss
        
        loss_dict['total'] = total_loss
        
        return loss_dict


if __name__ == '__main__':
    # Unit tests
    print("Testing loss functions...")
    
    batch_size = 4
    H, W = 352, 352
    
    # Create dummy data
    pred = torch.randn(batch_size, 1, H, W)
    target = torch.randint(0, 2, (batch_size, 1, H, W)).float()
   
    # Test Focal + IoU Loss
    print("\n1. Testing FocalIoULoss...")
    focal_iou_loss = FocalIoULoss()
    loss_value = focal_iou_loss(pred, target)
    print(f"   Loss value: {loss_value.item():.4f}")
    assert loss_value >= 0, "Loss should be non-negative"
    print("   ✓ FocalIoULoss works!")
    
    # Test Deep Supervision Loss  
    print("\n2. Testing DeepSupervisionLoss...")
    ds_loss = DeepSupervisionLoss()
    
    # Create multi-scale outputs
    outputs = {
        'main': pred,
        'aux_1_8': torch.randn(batch_size, 1, H//8, W//8),
        'aux_1_16': torch.randn(batch_size, 1, H//16, W//16),
        'aux_1_32': torch.randn(batch_size, 1, H//32, W//32),
    }
    
    loss_dict = ds_loss(outputs, target)
    
    print(f"   Total loss: {loss_dict['total'].item():.4f}")
    print(f"   Main loss: {loss_dict['main'].item():.4f}")
    print(f"   Aux 1/8 loss: {loss_dict['aux_1_8'].item():.4f}")
    print(f"   Aux 1/16 loss: {loss_dict['aux_1_16'].item():.4f}")
    print(f"   Aux 1/32 loss: {loss_dict['aux_1_32'].item():.4f}")
    
    assert loss_dict['total'] >= 0, "Total loss should be non-negative"
    print("   ✓ DeepSupervisionLoss works!")
    
    # Test backward
    loss_dict['total'].backward()
    print("\n✓ All loss functions passed!")
