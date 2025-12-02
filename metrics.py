"""
Evaluation Metrics for Polyp Segmentation

Implements:
- IoU (Intersection over Union / Jaccard Index)
- Dice Score (F1 Score)
- Precision
- Recall
- Specificity
- Mean Absolute Error (MAE)
"""

import torch
import numpy as np


def calculate_iou(pred, target, threshold=0.5, smooth=1e-6):
    """
    Calculate IoU (Intersection over Union / Jaccard Index).
    
    IoU = |A ∩ B| / |A ∪ B|
    
    Args:
        pred (Tensor or ndarray): Predictions, shape [B, 1, H, W] or [B, H, W]
        target (Tensor or ndarray): Ground truth, shape [B, 1, H, W] or [B, H, W]
        threshold (float): Threshold for binarization (default: 0.5)
        smooth (float): Smoothing factor to avoid division by zero
    
    Returns:
        float: Mean IoU across batch
    """
    # Convert to tensor if numpy
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    
    # Ensure pred is in range [0, 1]  (apply sigmoid if needed)
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)
    
    # Binarize predictions
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    # Flatten spatial dimensions
    pred_flat = pred_binary.view(pred_binary.size(0), -1)
    target_flat = target_binary.view(target_binary.size(0), -1)
    
    # Calculate intersection and union per sample
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
    
    # FIX: Properly handle empty cases
    # Only calculate IoU where union > 0
    iou = torch.zeros_like(intersection, dtype=torch.float32)
    valid_mask = (union > 0)
    
    if valid_mask.any():
        iou[valid_mask] = intersection[valid_mask] / union[valid_mask]
    
    # For empty predictions AND empty targets: return 0
    # (no positive class detected, should not be rewarded)
    # This penalizes models that predict nothing when there's nothing to predict
    
    return iou.mean().item()


def calculate_dice(pred, target, threshold=0.5, smooth=1e-6):
    """
    Calculate Dice Score (F1 Score).
    
    Dice = 2*|A ∩ B| / (|A| + |B|)
    
    Args:
        pred (Tensor or ndarray): Predictions, shape [B, 1, H, W] or [B, H, W]
        target (Tensor or ndarray): Ground truth, shape [B, 1, H, W] or [B, H, W]
        threshold (float): Threshold for binarization (default: 0.5)
        smooth (float): Smoothing factor
    
    Returns:
        float: Mean Dice score across batch
    """
    # Convert to tensor if numpy
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    
    # Ensure pred is in range [0, 1]
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)
    
    # Binarize predictions
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    # Flatten
    pred_flat = pred_binary.view(pred_binary.size(0), -1)
    target_flat = target_binary.view(target_binary.size(0), -1)
    
    # Calculate intersection
    intersection = (pred_flat * target_flat).sum(dim=1)
    
    # Calculate denominator
    denominator = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    
    # FIX: Properly handle empty cases
    dice = torch.zeros_like(intersection, dtype=torch.float32)
    valid_mask = (denominator > 0)
    
    if valid_mask.any():
        dice[valid_mask] = (2. * intersection[valid_mask]) / denominator[valid_mask]
    
    # Empty cases get dice=0
    
    return dice.mean().item()


def calculate_precision(pred, target, threshold=0.5, smooth=1e-6):
    """
    Calculate Precision.
    
    Precision = TP / (TP + FP)
    
    Args:
        pred (Tensor or ndarray): Predictions
        target (Tensor or ndarray): Ground truth
        threshold (float): Threshold for binarization
        smooth (float): Smoothing factor
    
    Returns:
        float: Mean precision across batch
    """
    # Convert to tensor if numpy
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    
    # Ensure pred is in range [0, 1]
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)
    
    # Binarize
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    # Flatten
    pred_flat = pred_binary.view(pred_binary.size(0), -1)
    target_flat = target_binary.view(target_binary.size(0), -1)
    
    # True Positives and False Positives
    tp = (pred_flat * target_flat).sum(dim=1)
    fp = (pred_flat * (1 - target_flat)).sum(dim=1)
    
    # FIX: Properly handle empty cases
    precision = torch.zeros_like(tp, dtype=torch.float32)
    denominator = tp + fp
    valid_mask = (denominator > 0)
    
    if valid_mask.any():
        precision[valid_mask] = tp[valid_mask] / denominator[valid_mask]
    
    # Empty predictions get precision=0
    
    return precision.mean().item()


def calculate_recall(pred, target, threshold=0.5, smooth=1e-6):
    """
    Calculate Recall (Sensitivity).
    
    Recall = TP / (TP + FN)
    
    Args:
        pred (Tensor or ndarray): Predictions
        target (Tensor or ndarray): Ground truth
        threshold (float): Threshold for binarization
        smooth (float): Smoothing factor
    
    Returns:
        float: Mean recall across batch
    """
    # Convert to tensor if numpy
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    
    # Ensure pred is in range [0, 1]
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)
    
    # Binarize
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    # Flatten
    pred_flat = pred_binary.view(pred_binary.size(0), -1)
    target_flat = target_binary.view(target_binary.size(0), -1)
    
    # True Positives and False Negatives
    tp = (pred_flat * target_flat).sum(dim=1)
    fn = ((1 - pred_flat) * target_flat).sum(dim=1)
    
    # FIX: Properly handle empty cases
    recall = torch.zeros_like(tp, dtype=torch.float32)
    denominator = tp + fn
    valid_mask = (denominator > 0)
    
    if valid_mask.any():
        recall[valid_mask] = tp[valid_mask] / denominator[valid_mask]
    
    # Empty GT get recall=0
    
    return recall.mean().item()


def calculate_specificity(pred, target, threshold=0.5, smooth=1e-6):
    """
    Calculate Specificity.
    
    Specificity = TN / (TN + FP)
    
    Args:
        pred (Tensor or ndarray): Predictions
        target (Tensor or ndarray): Ground truth
        threshold (float): Threshold for binarization
        smooth (float): Smoothing factor
    
    Returns:
        float: Mean specificity across batch
    """
    # Convert to tensor if numpy
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    
    # Ensure pred is in range [0, 1]
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)
    
    # Binarize
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    # Flatten
    pred_flat = pred_binary.view(pred_binary.size(0), -1)
    target_flat = target_binary.view(target_binary.size(0), -1)
    
    # True Negatives and False Positives
    tn = ((1 - pred_flat) * (1 - target_flat)).sum(dim=1)
    fp = (pred_flat * (1 - target_flat)).sum(dim=1)
    
    # FIX: Properly handle empty cases
    specificity = torch.zeros_like(tn, dtype=torch.float32)
    denominator = tn + fp
    valid_mask = (denominator > 0)
    
    if valid_mask.any():
        specificity[valid_mask] = tn[valid_mask] / denominator[valid_mask]
    
    # Empty negative class get specificity=0
    
    return specificity.mean().item()


def calculate_mae(pred, target):
    """
    Calculate Mean Absolute Error.
    
    MAE = mean(|pred - target|)
    
    Args:
        pred (Tensor or ndarray): Predictions (probabilities)
        target (Tensor or ndarray): Ground truth
    
    Returns:
        float: MAE
    """
    # Convert to tensor if numpy
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    
    # Ensure pred is in range [0, 1]
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)
    
    # Calculate MAE
    mae = torch.abs(pred - target).mean()
    
    return mae.item()


class MetricsCalculator:
    """
    Wrapper class for calculating all metrics at once.
    """
    
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    
    def __call__(self, pred, target):
        """
        Calculate all metrics.
        
        Args:
            pred (Tensor): Predictions
            target (Tensor): Ground truth
        
        Returns:
            dict: All metrics
        """
        metrics = {
            'iou': calculate_iou(pred, target, self.threshold),
            'dice': calculate_dice(pred, target, self.threshold),
            'precision': calculate_precision(pred, target, self.threshold),
            'recall': calculate_recall(pred, target, self.threshold),
            'specificity': calculate_specificity(pred, target, self.threshold),
            'mae': calculate_mae(pred, target),
        }
        
        return metrics


if __name__ == '__main__':
    # Unit tests
    print("Testing metrics...")
    
    # Create dummy data
    batch_size = 4
    H, W = 256, 256
    
    # Perfect prediction case
    print("\n1. Perfect prediction (all metrics should be ~1.0 or ~0.0 for MAE):")
    pred = torch.ones(batch_size, 1, H, W) * 0.9
    target = torch.ones(batch_size, 1, H, W)
    
    calc = MetricsCalculator(threshold=0.5)
    metrics = calc(pred, target)
    
    for name, value in metrics.items():
        print(f"   {name}: {value:.4f}")
    
    assert metrics['iou'] > 0.99, "IoU should be ~1.0 for perfect prediction"
    assert metrics['dice'] > 0.99, "Dice should be ~1.0 for perfect prediction"
    
    # Random prediction case
    print("\n2. Random prediction:")
    pred = torch.sigmoid(torch.randn(batch_size, 1, H, W))
    target = torch.randint(0, 2, (batch_size, 1, H, W)).float()
    
    metrics = calc(pred, target)
    
    for name, value in metrics.items():
        print(f"   {name}: {value:.4f}")
    
    assert 0 <= metrics['iou'] <= 1, "IoU should be in [0, 1]"
    assert 0 <= metrics['dice'] <= 1, "Dice should be in [0, 1]"
    
    print("\n✓ All metrics tests passed!")
