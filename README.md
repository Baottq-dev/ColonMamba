# ColonFormer: An Efficient Transformer based Method for Colon Polyp Segmentation

This repository contains the official Pytorch implementation of training & evaluation code for ColonFormer.

> **📌 Important Note - VMamba Integration (NEW!)**  
> This codebase now supports **VMamba** backbone with **SS2D** (Selective Scan 2D) attention for improved performance. All original SegFormer functionality remains fully supported.

---

## 🚀 New Features

- ✅ **VMamba Backbone** - State Space Model for vision
- ✅ **SS2D Attention** - Replace Axial Attention with Mamba-style scanning
- ✅ **Hybrid Mode** - Use SegFormer or VMamba backbone
- ✅ **MMSegmentation v1.x** + **PyTorch 2.x** support

---

## Environment Setup

### Conda + pip (Recommended)

```bash
# 1. Create environment
conda create -n colonformer python=3.10 -y
conda activate colonformer

# 2. Install CUDA toolkit (required for mamba-ssm compilation)
conda install cuda-toolkit=11.8 -c nvidia -y

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('✅ mamba-ssm OK')"
```


---

## Dataset

Download datasets:

1. **Training dataset**: [Google Drive](https://drive.google.com/file/d/1lODorfB33jbd-im-qrtUgWnZXxB94F55/view) → `./data/TrainDataset/`
2. **Testing dataset**: [Google Drive](https://drive.google.com/file/d/1o8OfBvYE6K-EpDyvzsmMPndnUMwb540R/view) → `./data/TestDataset/`

---

## Pretrained Weights

### SegFormer (MiT)
Download from [Google Drive](https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia?usp=sharing) → `pretrained/`

### VMamba
```bash
# Download VMamba pretrained weights
python download_vmamba_weights.py --model tiny
python download_vmamba_weights.py --model small
python download_vmamba_weights.py --model base
```

---

## Training

### SegFormer Backbone (Original)

```bash
python train.py --backbone b3 --train_path ./data/TrainDataset --train_save ColonFormer_B3
```

### VMamba Backbone (NEW!)

```bash
# VMamba + Axial Attention
python train.py \
    --backbone vmamba_small \
    --train_path ./data/TrainDataset \
    --train_save ColonFormer_VMamba_AA

# VMamba + SS2D Attention (Best!)
python train.py \
    --backbone vmamba_small \
    --use_ss2d \
    --train_path ./data/TrainDataset \
    --train_save ColonFormer_VMamba_SS2D
```

### Available Backbones

| Backbone | Type | Channels | Pretrained |
|----------|------|----------|------------|
| `b0` - `b5` | SegFormer (MiT) | [32-64, 64-128, 160-320, 256-512] | ✅ |
| `vmamba_tiny` | VMamba | [96, 192, 384, 768] | ✅ |
| `vmamba_small` | VMamba | [96, 192, 384, 768] | ✅ |
| `vmamba_base` | VMamba | [128, 256, 512, 1024] | ✅ |

---

## Evaluation

```bash
python test.py --backbone b3 --weight ./snapshots/ColonFormerB3/last.pth --test_path ./data/TestDataset
```

---

## Changelog

### v2.x - VMamba Integration (2024)
- ✅ Added **VMamba** backbone support (Tiny, Small, Base)
- ✅ Added **SS2D** attention module for spatial modeling
- ✅ Dynamic channel detection for different backbones
- ✅ Pretrained weights auto-download script
- ✅ Updated requirements with mamba-ssm, triton dependencies

### v1.x - MMSegmentation Upgrade (2024)
- ✅ Upgraded to MMSegmentation v1.x and PyTorch 2.x
- ✅ Fixed pretrained weights loading compatibility
- ✅ Optimized training speed

---

## Citation

If you find this code useful in your research, please consider citing:

```bibtex
@article{duc2022colonformer,
  title={Colonformer: An efficient transformer based method for colon polyp segmentation},
  author={Duc, Nguyen Thanh and Oanh, Nguyen Thi and Thuy, Nguyen Thi and Triet, Tran Minh and Dinh, Viet Sang},
  journal={IEEE Access},
  volume={10},
  pages={80575--80586},
  year={2022},
  publisher={IEEE}
}
```

```bibtex
@inproceedings{liu2024vmamba,
  title={VMamba: Visual State Space Model},
  author={Liu, Yue and Tian, Yunjie and Zhao, Yuzhong and Yu, Hongtian and Xie, Lingxi and Wang, Yaowei and Ye, Qixiang and Liu, Yunfan},
  booktitle={arXiv preprint arXiv:2401.10166},
  year={2024}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
