"""
Dataset Loader for Polyp Segmentation

Supports:
- Training dataset with images and masks
- Multiple test datasets (Kvasir, CVC-ClinicDB, ETIS, etc.)
- Data augmentation for training
- Normalization
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PolypDataset(Dataset):
    """
    Polyp segmentation dataset.
    
    Args:
        image_dir (str): Directory containing images
        mask_dir (str): Directory containing masks
        transform (albumentations.Compose): Augmentation pipeline
        image_size (int): Target image size (default: 352)
    """
    
    def __init__(self, image_dir, mask_dir, transform=None, image_size=352):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_size = image_size
        
        # Get all image files
        self.images = sorted([f for f in os.listdir(image_dir) 
                             if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))])
        
        print(f"Loaded {len(self.images)} images from {image_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict: {'image': Tensor [3, H, W], 'mask': Tensor [1, H, W], 'name': str}
        """
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask (handle different mask naming conventions)
        mask_name = self._get_mask_name(img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise IOError(
                    f"Failed to read mask file: {mask_path}\n"
                    f"File exists but may be corrupted or in unsupported format."
                )
        else:
            # Try different extensions
            mask = None
            for ext in ['.png', '.jpg', '.bmp', '.tif']:
                mask_path_try = os.path.join(self.mask_dir, 
                                             os.path.splitext(img_name)[0] + ext)
                if os.path.exists(mask_path_try):
                    mask = cv2.imread(mask_path_try, cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        raise IOError(
                            f"Failed to read mask file: {mask_path_try}\n"
                            f"File exists but may be corrupted or in unsupported format."
                        )
                    break
            
            if mask is None:
                raise FileNotFoundError(f"Mask not found for {img_name}")
        
        # Binarize mask (0 or 255 -> 0 or 1)
        mask = (mask > 127).astype(np.uint8)
        
        # Apply transforms
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Default: resize and normalize
            image = cv2.resize(image, (self.image_size, self.image_size))
            mask = cv2.resize(mask, (self.image_size, self.image_size), 
                            interpolation=cv2.INTER_NEAREST)
            
            # To tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return {
            'image': image,
            'mask': mask,
            'name': img_name
        }
    
    def _get_mask_name(self, img_name):
        """Handle different mask naming conventions"""
        base_name = os.path.splitext(img_name)[0]
        
        # Try common mask naming patterns
        possible_names = [
            img_name,  # Same name as image
            base_name + '.png',
            base_name + '.jpg',
            base_name + '_mask.png',
            base_name + '_mask.jpg',
        ]
        
        return possible_names[0]


def get_train_transform(image_size=352):
    """
    Training transform WITHOUT augmentation (match ColonFormer baseline).
    Only resize and normalize.
    
    For augmentation experiments, use get_train_transform_with_aug() instead.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_train_transform_with_aug(image_size=352):
    """
    Training transform WITH augmentation (for later tuning).
    
    Includes:
    - Random horizontal/vertical flip
    - Random rotation
    - Random brightness/contrast
    - Elastic transform
    - Grid distortion
    - Normalization
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=45,
            p=0.5
        ),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
        ], p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.GaussianBlur(blur_limit=3, p=0.1),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_val_transform(image_size=352):
    """
    Validation/test transform (no augmentation).
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_dataloaders(
    train_img_dir,
    train_mask_dir,
    val_img_dir=None,
    val_mask_dir=None,
    batch_size=16,
    num_workers=4,
    image_size=352,
    val_split=0.1,
):
    """
    Create train and validation dataloaders.
    
    Args:
        train_img_dir (str): Training images directory
        train_mask_dir (str): Training masks directory
        val_img_dir (str): Validation images directory (optional)
        val_mask_dir (str): Validation masks directory (optional)
        batch_size (int): Batch size
        num_workers (int): Number of dataloader workers
        image_size (int): Image size
        val_split (float): Validation split ratio if val_dir not provided
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Training dataset
    train_dataset = PolypDataset(
        image_dir=train_img_dir,
        mask_dir=train_mask_dir,
        transform=get_train_transform(image_size),
        image_size=image_size,
    )
    
    # Validation dataset
    if val_img_dir is not None and val_mask_dir is not None:
        # Use separate validation set
        val_dataset = PolypDataset(
            image_dir=val_img_dir,
            mask_dir=val_mask_dir,
            transform=get_val_transform(image_size),
            image_size=image_size,
        )
    else:
        # Split from training set
        total_size = len(train_dataset)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Update val_dataset transform to remove augmentation
        val_dataset.dataset.transform = get_val_transform(image_size)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    return train_loader, val_loader


def get_test_dataloader(
    test_img_dir,
    test_mask_dir,
    batch_size=1,
    num_workers=2,
    image_size=352,
):
    """
    Create test dataloader (no augmentation).
    
    Args:
        test_img_dir (str): Test images directory
        test_mask_dir (str): Test masks directory
        batch_size (int): Batch size (default: 1 for testing)
        num_workers (int): Number of workers
        image_size (int): Image size
    
    Returns:
        DataLoader: Test dataloader
    """
    test_dataset = PolypDataset(
        image_dir=test_img_dir,
        mask_dir=test_mask_dir,
        transform=get_val_transform(image_size),
        image_size=image_size,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    return test_loader


if __name__ == '__main__':
    # Test dataloader
    print("Testing PolypDataset...")
    
    # Paths
    train_img_dir = 'data/TrainDataset/image'
    train_mask_dir = 'data/TrainDataset/mask'
    
    # Create dataset
    dataset = PolypDataset(
        image_dir=train_img_dir,
        mask_dir=train_mask_dir,
        transform=get_train_transform(352),
        image_size=352,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    print(f"Image range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
    print(f"Mask range: [{sample['mask'].min():.3f}, {sample['mask'].max():.3f}]")
    print(f"Name: {sample['name']}")
    
    # Test dataloader
    print("\nTesting DataLoader...")
    train_loader, val_loader = get_dataloaders(
        train_img_dir=train_img_dir,
        train_mask_dir=train_mask_dir,
        batch_size=4,
        num_workers=0,
        image_size=352,
        val_split=0.1,
    )
    
    # Get one batch
    batch = next(iter(train_loader))
    print(f"Batch images shape: {batch['image'].shape}")
    print(f"Batch masks shape: {batch['mask'].shape}")
    
    print("\n✓ Dataset and DataLoader test passed!")
