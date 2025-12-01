# ColonMamba - Hybrid ResNet-VMamba for Polyp Segmentation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.10+](https://img.shields.io/badge/pytorch-1.10+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**State-of-the-art polyp segmentation** combining ResNet's local feature extraction with VMamba's efficient global context modeling.

## Architecture Overview

```
Input → ResNet-34 (Stages 1-2) → FTM Bridge → VMamba-Tiny (Stages 3-4) → PPD Decoder + MRR → Segmentation Mask
        Local Details (CNN)        384ch         Global Context (Mamba)      Multi-scale Refinement
```

**Key Innovations:**
- 🔥 **Hybrid Encoder**: ResNet (local) + VMamba (global) for better feature extraction
- ⚡ **Linear Complexity**: O(N) vs O(N²) for transformer-based models  
- 🎯 **MRR Refinement**: Mamba Reverse Refinement for sharp boundary delineation
- 📊 **Deep Supervision**: Multi-scale loss at 4 output levels

**Model Size**: ~40-50M parameters | **Speed**: ~30-50 FPS on RTX 3090

---

## Installation

### Requirements
```bash
python >= 3.10
torch 
torchvision
opencv-python
albumentations
tensorboard
tqdm
einops
matplotlib
Pillow
```

### Setup
```bash
# Clone repository
git clone https://github.com/Baottq-dev/ColonMamba.git
cd ColonMamba

# Install dependencies
pip install -r requirements.txt

# Download VMamba pretrained weights (REQUIRED for best performance)
# Download: vssm_tiny_0230_ckpt_epoch_292.pth from MzeroMiko/VMamba
# Place in: checkpoints/vmamba/vssm_tiny_0230_ckpt_epoch_292.pth
mkdir -p checkpoints/vmamba
# wget <vmamba_checkpoint_url> -O checkpoints/vmamba/vssm_tiny_0230_ckpt_epoch_292.pth
```

---

## Dataset Preparation

Expected directory structure:
```
data/
├── TrainDataset/
│   ├── image/
│   └── mask/
└── TestDataset/
    ├── Kvasir/
    │   ├── images/
    │   └── masks/
    ├── CVC-ClinicDB/
    │   ├── images/
    │   └── masks/
    └── ETIS-LaribPolypDB/
        ├── images/
        └── masks/
```

**Supported Datasets:**
- [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/)
- [CVC-ClinicDB](https://polyp.grand-challenge.org/CVCClinicDB/)
- [ETIS-LaribPolypDB](https://polyp.grand-challenge.org/EtisLarib/)
- [CVC-ColonDB](http://mv.cvc.uab.es/projects/colon-qa/cvccolondb)

---

## Training

**Note**: Both ResNet-34 (ImageNet) and VMamba-Tiny are pretrained by default for optimal performance.

### Standard Training (with both pretrained weights)
```bash
python train.py \
    --train_img_dir data/TrainDataset/image \
    --train_mask_dir data/TrainDataset/mask \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-4 \
    --save_dir checkpoints/experiment1
```

### Training with Mixed Precision (Recommended for faster training)
```bash
python train.py \
    --train_img_dir data/TrainDataset/image \
    --train_mask_dir data/TrainDataset/mask \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-4 \
    --mixed_precision \
    --save_dir checkpoints/experiment_fp16
```

### Full Training Arguments
```
Data:
  --train_img_dir       Training images directory
  --train_mask_dir      Training masks directory
  --val_split           Validation split ratio (default: 0.1)
  --image_size          Input size (default: 352)

Model:
  --pretrained_resnet   Use ImageNet pretrained ResNet-34 (default: True)
  --vmamba_ckpt         Path to VMamba pretrained checkpoint (REQUIRED, default: checkpoints/vmamba/vssm_tiny_0230_ckpt_epoch_292.pth)
  --use_mrr             Enable Mamba Reverse Refinement (default: True)
  --decoder_channels    Decoder base channels (default: 256)

**Note**: VMamba pretrained weights are MANDATORY for this architecture. The model will not work without them.

Training:
  --epochs              Number of epochs (default: 100)
  --batch_size          Batch size (default: 16)
  --lr                  Learning rate (default: 1e-4)
  --weight_decay        Weight decay (default: 1e-4)
  --aux_weight          Auxiliary loss weight (default: 0.4)
  --mixed_precision     Enable mixed precision (FP16)

System:
  --num_workers         DataLoader workers (default: 4)
  --save_dir            Checkpoint directory
  --save_freq           Save every N epochs (default: 10)
  --resume              Resume from checkpoint
```

---

## Inference

### Single Image
```bash
python inference.py \
    --mode single \
    --checkpoint checkpoints/experiment1/checkpoint_epoch100_best.pth \
    --image path/to/image.jpg \
    --output results/ \
    --visualize
```

### Batch Evaluation on Test Dataset
```bash
python inference.py \
    --mode dataset \
    --checkpoint checkpoints/experiment1/checkpoint_epoch100_best.pth \
    --test_img_dir data/TestDataset/Kvasir/images \
    --test_mask_dir data/TestDataset/Kvasir/masks \
    --batch_size 4 \
    --output results/kvasir/ \
    --visualize
```

**Output Metrics:**
- IoU (Intersection over Union)
- Dice Score (F1)
- Precision
- Recall
- Specificity
- MAE (Mean Absolute Error)

---

## Project Structure

```
ColonMamba/
├── models/
│   ├── __init__.py
│   ├── ftm_bridge.py          # Feature Transition Module
│   ├── vmamba_utils.py        # SS2D cross-scan implementation
│   ├── mrr.py                 # Mamba Reverse Refinement
│   ├── hybrid_encoder.py      # ResNet-VMamba encoder
│   ├── decoder.py             # PPD decoder
│   └── colonmamba.py          # Main model
├── dataset.py                 # Data loading & augmentation
├── losses.py                  # Loss functions
├── metrics.py                 # Evaluation metrics
├── train.py                   # Training script
├── inference.py               # Inference & evaluation
├── utils.py                   # Utilities
└── README.md
```

---

## Results

### Performance on Standard Benchmarks

| Dataset | IoU | Dice | Precision | Recall |
|---------|-----|------|-----------|--------|
| Kvasir-SEG | TBD | TBD | TBD | TBD |
| CVC-ClinicDB | TBD | TBD | TBD | TBD |
| ETIS | TBD | TBD | TBD | TBD |

*Results will be updated after training on standard benchmarks*

---

## Key Components

### 1. FTM Bridge
Semantic projection from ResNet features (128ch) to VMamba space (384ch) using Conv1x1 + BatchNorm + GELU.

### 2. SS2D (Selective Scan 2D)
4-directional cross-scanning (↘, ↙, ↗, ↖) for capturing truly global spatial dependencies at linear complexity.

### 3. MRR (Mamba Reverse Refinement)
Replaces axial attention with Mamba cross-scan for boundary refinement:
- Reverse masking focuses on uncertain regions
- 4-directional scanning preserves cross-spatial correlation
- O(N) complexity vs O(N²) for attention

### 4. Deep Supervision
Multi-scale outputs at 4 levels (main + auxiliary at 1/8, 1/16, 1/32) with weighted loss combination.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{colonmamba2025,
  title={ColonMamba: Hybrid ResNet-VMamba Architecture for Polyp Segmentation},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

---

## Acknowledgments

- **VMamba**: [MzeroMiko/VMamba](https://github.com/MzeroMiko/VMamba)
- **ResNet**: torchvision implementation
- **ColonFormer**: Original architecture inspiration

---

## License

This project is licensed under the MIT License - see LICENSE file for details.

---

## Contact

For questions or collaboration:
- Email: baottqdeveloper@gmail.com
- GitHub Issues: [Submit an issue](https://github.com/Baottq-dev/ColonMamba/issues)

---

**Note**: This is a research implementation. For clinical use, please ensure proper validation and regulatory approval.
