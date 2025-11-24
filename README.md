# ColonFormer: An Efficient Transformer based Method for Colon Polyp Segmentation
This repository contains the official Pytorch implementation of training & evaluation code for ColonFormer.

> **📌 Important Note - MMSegmentation v1.x Upgrade**  
> This codebase has been upgraded to use **MMSegmentation v1.x** and **PyTorch 2.x** for improved performance and compatibility. All core functionality remains equivalent to the original implementation while providing faster training speeds.

### Environment
- Creating a virtual environment in terminal: `conda create -n ColonFormer python=3.8`
- Install PyTorch 2.x with CUDA support:
  ```bash
  conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
  ```
- Install other requirements: `pip install -r requirements.txt`

### Dataset
Downloading necessary data:
1. For `Experiment 1` in our paper: 
    - Download testing dataset and move it into `./data/TestDataset/`, which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1o8OfBvYE6K-EpDyvzsmMPndnUMwb540R/view).
    - Download training dataset and move it into `./data/TrainDataset/`, which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1lODorfB33jbd-im-qrtUgWnZXxB94F55/view).
2. For `Experiment 2` and `Experiment 3`:
    - All datasets we use in this experiments can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1ExJeVqbcBn6yy-gdGqEYw5phJywHIUXZ/view?usp=sharing)
    
### Training

#### Download Pretrained Weights

**Option 1: Automatic Download (Recommended)**
```bash
python download_all_weights.py
```
This will automatically download all MiT pretrained weights (b0-b5) compatible with MMSegmentation v1.x from the official OpenMMLab repository.

**Option 2: Manual Download**
- Download official MMSegmentation v1.x compatible weights from [OpenMMLab](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/segformer#pretrained-models)
- Or use original weights from [NVLabs SegFormer](https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia?usp=sharing)
- Put them in `pretrained/` folder

#### Run Training
```bash
python train.py --backbone b3 --num_epochs 20 --batchsize 8 --train_path ./data/TrainDataset --train_save ColonFormerB3
```

**Available backbones**: `b0`, `b1`, `b2`, `b3`, `b4`, `b5`

**Performance Note**: Training on PyTorch 2.x with optimized settings provides significantly faster iteration times compared to PyTorch 1.x.

### Evaluation
For evaluation, specific your backbone version, weight's path and dataset and run `test.py`. For example:
```bash
python test.py --backbone b3 --weight ./snapshots/ColonFormerB3/last.pth --test_path ./data/TestDataset
```
We provide some [pretrained weights](https://drive.google.com/drive/folders/1SVxluPlRVohkN6Q6hG-FpA9L8eapZuxa?usp=sharing) in case you need.

### Changelog

#### v1.x - MMSegmentation Upgrade (2024)
- ✅ Upgraded to **MMSegmentation v1.x** and **PyTorch 2.x**
- ✅ Fixed pretrained weights loading compatibility
- ✅ Optimized training speed (disabled anomaly detection for production use)
- ✅ Added automatic pretrained weights download script (`download_all_weights.py`)
- ✅ Updated data loading paths for consistency
- ✅ Maintained 100% functional equivalence with original implementation
- ✅ All neural network architectures remain identical

### Citation
If you find this code useful in your research, please consider citing:

```
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

```
@inproceedings{ngoc2021neounet,
  title={NeoUNet: Towards accurate colon polyp segmentation and neoplasm detection},
  author={Ngoc Lan, Phan and An, Nguyen Sy and Hang, Dao Viet and Long, Dao Van and Trung, Tran Quang and Thuy, Nguyen Thi and Sang, Dinh Viet},
  booktitle={Advances in Visual Computing: 16th International Symposium, ISVC 2021, Virtual Event, October 4-6, 2021, Proceedings, Part II},
  pages={15--28},
  year={2021},
  organization={Springer}
}
```

```
@article{thuan2023rabit,
  title={RaBiT: An Efficient Transformer using Bidirectional Feature Pyramid Network with Reverse Attention for Colon Polyp Segmentation},
  author={Thuan, Nguyen Hoang and Oanh, Nguyen Thi and Thuy, Nguyen Thi and Perry, Stuart and Sang, Dinh Viet},
  journal={arXiv preprint arXiv:2307.06420},
  year={2023}
}
```
