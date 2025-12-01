"""
Utility functions for ColonMamba

Contains helper functions for:
- Visualization
- Model parameter counting
- Config management
- Logging
"""

import os
import random
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from matplotlib import colors


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """Count trainable and total parameters"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def get_lr(optimizer):
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def create_color_map():
    """Create custom colormap for visualization"""
    cmap = colors.ListedColormap(['black', 'red', 'green', 'yellow'])
    bounds = [0, 1, 2, 3, 4]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def overlay_mask(image, mask, alpha=0.5, color=(0, 255, 0)):
    """
    Overlay mask on image.
    
    Args:
        image (ndarray): RGB image [H, W, 3]
        mask (ndarray): Binary mask [H, W]
        alpha (float): Transparency
        color (tuple): Mask color in RGB
    
    Returns:
        ndarray: Image with overlaid mask
    """
    overlay = image.copy()
    mask_3ch = np.stack([mask] * 3, axis=-1)
    
    # Apply color to mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color
    
    # Blend
    overlay = cv2.addWeighted(overlay, 1-alpha, colored_mask, alpha, 0)
    
    return overlay


def save_comparison(image, gt_mask, pred_mask, save_path, threshold=0.5):
    """
    Save side-by-side comparison of GT and prediction.
    
    Args:
        image (ndarray): Original image [H, W, 3]
        gt_mask (ndarray): Ground truth [H, W]
        pred_mask (ndarray): Prediction [0, 1] [H, W]
        save_path (str): Output path
        threshold (float): Binarization threshold
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Images
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontsize=14)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gt_mask, cmap='gray')
    axes[0, 1].set_title('Ground Truth', fontsize=14)
    axes[0, 1].axis('off')
    
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    axes[0, 2].imshow(pred_binary, cmap='gray')
    axes[0, 2].set_title(f'Prediction (thresh={threshold})', fontsize=14)
    axes[0, 2].axis('off')
    
    # Row 2: Overlays
    gt_overlay = overlay_mask(image, gt_mask, alpha=0.4, color=(255, 0, 0))
    axes[1, 0].imshow(gt_overlay)
    axes[1, 0].set_title('GT Overlay (Red)', fontsize=14)
    axes[1, 0].axis('off')
    
    pred_overlay = overlay_mask(image, pred_binary, alpha=0.4, color=(0, 255, 0))
    axes[1, 1].imshow(pred_overlay)
    axes[1, 1].set_title('Pred Overlay (Green)', fontsize=14)
    axes[1, 1].axis('off')
    
    # Combined overlay
    combined = image.copy()
    combined[gt_mask > 0] = [255, 0, 0]  # GT in red
    combined[pred_binary > 0] = [0, 255, 0]  # Pred in green
    overlap = (gt_mask > 0) & (pred_binary > 0)
    combined[overlap] = [255, 255, 0]  # Overlap in yellow
    
    axes[1, 2].imshow(combined)
    axes[1, 2].set_title('Combined (Yellow=Overlap)', fontsize=14)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def calculate_confusion_matrix(pred, target, threshold=0.5):
    """
    Calculate confusion matrix components.
    
    Returns:
        dict: {'TP', 'TN', 'FP', 'FN'}
    """
    pred_binary = (pred > threshold).astype(np.uint8)
    target_binary = (target > 0).astype(np.uint8)
    
    TP = np.sum((pred_binary == 1) & (target_binary == 1))
    TN = np.sum((pred_binary == 0) & (target_binary == 0))
    FP = np.sum((pred_binary == 1) & (target_binary == 0))
    FN = np.sum((pred_binary == 0) & (target_binary == 1))
    
    return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    # Test utilities
    print("Testing utility functions...")
    
    # Test seed setting
    set_seed(42)
    print(f"Random number after seed: {random.random()}")
    
    # Test AverageMeter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"Average: {meter.avg}")
    
    print("✓ Utilities test passed!")
