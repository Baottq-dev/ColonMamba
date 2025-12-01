"""
ColonMamba - Hybrid ResNet-VMamba for Polyp Segmentation

A state-of-the-art deep learning architecture combining:
- ResNet-34 (local feature extraction)
- VMamba-Tiny (global context modeling)
- Mamba Reverse Refinement (efficient boundary refinement)
"""

__version__ = '1.0.0'
__author__ = 'ColonMamba Team'

# Main model
from .models import ColonMamba

# Data handling
from .dataset import get_dataloaders, get_test_dataloader, PolypDataset

# Loss functions
from .losses import (
    DeepSupervisionLoss,
    BCEDiceLoss,
    DiceLoss,
    FocalLoss,
)

# Metrics
from .metrics import MetricsCalculator

# Utilities
from .utils import (
    set_seed,
    count_parameters,
    save_predictions,
)

__all__ = [
    # Model
    'ColonMamba',
    
    # Data
    'get_dataloaders',
    'get_test_dataloader',
    'PolypDataset',
    
    # Losses
    'DeepSupervisionLoss',
    'BCEDiceLoss',
    'DiceLoss',
    'FocalLoss',
    
    # Metrics
    'MetricsCalculator',
    
    # Utils
    'set_seed',
    'count_parameters',
    'save_predictions',
]
