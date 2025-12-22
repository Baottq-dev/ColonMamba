import os
os.environ['MMENGINE_LOG_LEVEL'] = 'ERROR'
os.environ['MMCV_LOG_LEVEL'] = 'ERROR'

import argparse
import os
import random
import sys
import time
import numpy as np
import cv2
import csv
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from utils import clip_gradient, AvgMeter
from torch.autograd import Variable
from datetime import datetime
import torch.nn.functional as F

import albumentations as A
from albumentations import Compose, OneOf

from mmseg import __version__
from mmseg.models.segmentors import ColonFormer as UNet

import logging
import warnings
warnings.filterwarnings('ignore')

# Cấu hình backbone kiểu cũ nhưng theo API mới
BACKBONE_CONFIGS = {
    'b0': dict(
        type='MixVisionTransformer',
        embed_dims=32,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3)),
    'b1': dict(
        type='MixVisionTransformer',
        embed_dims=64,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3)),
    'b2': dict(
        type='MixVisionTransformer',
        embed_dims=64,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3)),
    'b3': dict(
        type='MixVisionTransformer',
        embed_dims=64,
        num_layers=[3, 4, 18, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3)),
    'b4': dict(
        type='MixVisionTransformer',
        embed_dims=64,
        num_layers=[3, 8, 27, 3],
        num_heads=[1, 2, 8, 16],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3)),
    'b5': dict(
        type='MixVisionTransformer',
        embed_dims=64,
        num_layers=[3, 6, 40, 3],
        num_heads=[1, 2, 10, 20],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3)),
    # VMamba backbones
    'vmamba_tiny': dict(
        type='Backbone_VSSM',
        dims=96,
        depths=[2, 2, 5, 2],
        out_indices=(0, 1, 2, 3),
        drop_path_rate=0.2,
        patch_size=4,
        norm_layer='ln2d',
        ssm_d_state=1,  # Must match pretrained checkpoint
        ssm_ratio=2.0,
        ssm_conv=3,
        ssm_conv_bias=False,
        forward_type='v05_noz',
        mlp_ratio=4.0,
        downsample_version='v3',
        patchembed_version='v2',
    ),
    'vmamba_small': dict(
        type='Backbone_VSSM',
        dims=96,
        depths=[2, 2, 15, 2],
        out_indices=(0, 1, 2, 3),
        drop_path_rate=0.3,
        patch_size=4,
        norm_layer='ln2d',
        ssm_d_state=1,  # Must match pretrained checkpoint
        ssm_ratio=2.0,
        ssm_conv=3,
        ssm_conv_bias=False,
        forward_type='v05_noz',
        mlp_ratio=4.0,
        downsample_version='v3',
        patchembed_version='v2',
    ),
    'vmamba_base': dict(
        type='Backbone_VSSM',
        dims=128,
        depths=[2, 2, 15, 2],
        out_indices=(0, 1, 2, 3),
        drop_path_rate=0.6,
        patch_size=4,
        norm_layer='ln2d',
        ssm_d_state=1,  # Must match pretrained checkpoint
        ssm_ratio=2.0,
        ssm_conv=3,
        ssm_conv_bias=False,
        forward_type='v05_noz',
        mlp_ratio=4.0,
        downsample_version='v3',
        patchembed_version='v2',
    ),
}

BACKBONE_CHANNELS = {
    'b0': [32, 64, 160, 256],
    'b1': [64, 128, 320, 512],
    'b2': [64, 128, 320, 512],
    'b3': [64, 128, 320, 512],
    'b4': [64, 128, 320, 512],
    'b5': [64, 128, 320, 512],
    # VMamba channels
    'vmamba_tiny': [96, 192, 384, 768],
    'vmamba_small': [96, 192, 384, 768],
    'vmamba_base': [128, 256, 512, 1024],
}

def get_backbone_cfg(name):
    if name not in BACKBONE_CONFIGS:
        raise ValueError(f'Unsupported backbone: {name}')
    return BACKBONE_CONFIGS[name], BACKBONE_CHANNELS[name]

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, img_paths, mask_paths, aug=True, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.aug = aug
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = cv2.resize(image, (352, 352))
            mask = cv2.resize(mask, (352, 352)) 

        image = image.astype('float32') / 255
        image = image.transpose((2, 0, 1))

        mask = mask[:,:,np.newaxis]
        mask = mask.astype('float32') / 255
        mask = mask.transpose((2, 0, 1))

        return np.asarray(image), np.asarray(mask)
    
epsilon = 1e-7

def recall_m(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    possible_positives = torch.sum(torch.round(torch.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + epsilon)
    return recall

def precision_m(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    predicted_positives = torch.sum(torch.round(torch.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + epsilon)
    return precision

def dice_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+epsilon))

def iou_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return recall*precision/(recall+precision-recall*precision + epsilon)


class FocalLossV1(nn.Module):
    
    def __init__(self,
                alpha=0.25,
                gamma=2,
                reduction='mean',):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        # compute loss
        logits = logits.float() # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)
        ce_loss = self.crit(logits, label.float())
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wfocal = FocalLossV1()(pred, mask)
    wfocal = (wfocal*weit).sum(dim=(2,3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wfocal + wiou).mean()


def train(train_loader, model, optimizer, epoch, lr_scheduler, args):
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record = AvgMeter()
    dice, iou = AvgMeter(), AvgMeter()
    # with torch.autograd.set_detect_anomaly(True):
    progress_bar = tqdm(
        train_loader,
        total=total_step,
        desc=f"Epoch {epoch}/{args.num_epochs}",
        leave=False
    )
    for i, pack in enumerate(progress_bar, start=1):
        if epoch <= 1:
                optimizer.param_groups[0]["lr"] = (epoch * i) / (1.0 * total_step) * args.init_lr
        else:
            lr_scheduler.step()

        for rate in size_rates: 
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            images = images.to(memory_format=torch.channels_last)  
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(args.init_trainsize*rate/32)*32)
            images = F.interpolate (images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            gts = F.interpolate (gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            map4, map3, map2, map1 = model(images)
            map1 = F.interpolate (map1, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            map2 = F.interpolate (map2, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            map3 = F.interpolate (map3, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            map4 = F.interpolate (map4, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            loss = structure_loss(map1, gts) + structure_loss(map2, gts) + structure_loss(map3, gts) + structure_loss(map4, gts)
            # with torch.autograd.set_detect_anomaly(True):
            #loss = nn.functional.binary_cross_entropy(map1, gts)
            # ---- metrics ----
            dice_score = dice_m(map4, gts)
            iou_score = iou_m(map4, gts)
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, args.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record.update(loss.data, args.batchsize)
                dice.update(dice_score.data, args.batchsize)
                iou.update(iou_score.data, args.batchsize)
                progress_bar.set_postfix({
                    "loss": float(loss_record.val),
                    "dice": float(dice.val),
                    "iou": float(iou.val)
                })

        # ---- train visualization ----
        if i == total_step:
            print('{} Training Epoch [{:03d}/{:03d}], '
                    '[loss: {:0.4f}, dice: {:0.4f}, iou: {:0.4f}]'.
                    format(datetime.now(), epoch, args.num_epochs,\
                            loss_record.show(), dice.show(), iou.show()))
    
    # Return metrics for best model tracking
    return loss_record.show(), dice.show(), iou.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int,
                        default=20, help='epoch number')
    parser.add_argument('--backbone', type=str,
                        default='b3', help='backbone version')
    parser.add_argument('--init_lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=4, help='training batch size')
    parser.add_argument('--init_trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--train_path', type=str,
                        default='./data/TrainDataset', help='path to train dataset')
    parser.add_argument('--train_save', type=str, default='auto',
                        help='Experiment name for saving (default: auto-generated from params)')
    parser.add_argument('--resume_path', type=str, help='path to checkpoint for resume training',
                        default='') 
    parser.add_argument('--pretrained_path', type=str, help='path to pretrained backbone weights',
                        default='')  
    parser.add_argument('--attention_type', type=str, default='ss2d',
                        choices=['aa_kernel', 'ss2d'],
                        help='Type of spatial attention: aa_kernel or ss2d (default: ss2d)')
    parser.add_argument('--use_local_global', action='store_true',
                        help='Enable 2-Branch Bottleneck (Local DW-Conv + Global Attention)')
    parser.add_argument('--freeze_epochs', type=int, default=0,
                        help='Freeze backbone for first N epochs (default: 0 = no freeze)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for AdamW optimizer (default: 0.01)')
    parser.add_argument('--backbone_lr', type=float, default=None,
                        help='Learning rate for backbone after unfreezing (default: init_lr/10)')
    args = parser.parse_args()
    
    # Set default backbone_lr if not provided
    if args.backbone_lr is None:
        args.backbone_lr = args.init_lr / 10

    # Auto-generate experiment name if not provided
    if args.train_save == 'auto':
        # Format: {backbone}_{attention}[_lg][_freeze{N}]_ep{epochs}
        name_parts = [args.backbone]
        
        # Attention type
        if args.use_local_global:
            name_parts.append(f'lg_{args.attention_type}')  # lg = local_global
        else:
            name_parts.append(args.attention_type)
        
        # Freeze epochs (if used)
        if args.freeze_epochs > 0:
            name_parts.append(f'freeze{args.freeze_epochs}')
        
        # Number of epochs
        name_parts.append(f'ep{args.num_epochs}')
        
        args.train_save = '_'.join(name_parts)
        print(f"[Auto] Experiment name: {args.train_save}")

    logging.getLogger('mmengine').setLevel(logging.WARNING)

    save_path = 'snapshots/{}/'.format(args.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    else:
        print("Save path existed")

    # Setup training log CSV
    log_path = os.path.join(save_path, 'training_log.csv')
    log_exists = os.path.exists(log_path)
    log_file = open(log_path, 'a', newline='')
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow(['epoch', 'loss', 'dice', 'iou', 'lr', 'best_iou', 'timestamp'])
        log_file.flush()

    train_img_paths = []
    train_mask_paths = []
    train_img_paths = glob('{}/image/*'.format(args.train_path))
    train_mask_paths = glob('{}/mask/*'.format(args.train_path))
    train_img_paths.sort()
    train_mask_paths.sort()

    train_dataset = Dataset(train_img_paths, train_mask_paths)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    total_step = len(train_loader)

    backbone_cfg, in_channels = get_backbone_cfg(args.backbone)
    
    model = UNet(
                backbone=backbone_cfg, 
                decode_head=dict(
                    type='UPerHead',
                    in_channels=in_channels,
                    in_index=[0, 1, 2, 3],
                    channels=128,
                    dropout_ratio=0.1,
                    num_classes=1,
                    norm_cfg=dict(type='BN', requires_grad=True),
                    align_corners=False,
                    loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
                neck=None,
                auxiliary_head=None,
                train_cfg=dict(),
                test_cfg=dict(mode='whole'),
                attention_type=args.attention_type,  # Spatial attention type: 'aa_kernel' or 'ss2d'
                use_local_global=args.use_local_global,  # Enable 2-Branch Bottleneck
                pretrained=args.pretrained_path if args.pretrained_path else (
                    'pretrained/vssm_{}.pth'.format(args.backbone.replace('vmamba_', '')) 
                    if 'vmamba' in args.backbone 
                    else 'pretrained/mit_{}.pth'.format(args.backbone)
                )).cuda()
    
    model = model.to(memory_format=torch.channels_last)

    # ========== FREEZE BACKBONE (if enabled) ==========
    if args.freeze_epochs > 0:
        model.freeze_backbone()
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    # ---- flops and params ----
    # Only pass trainable parameters to optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.init_lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                        T_max=len(train_loader)*args.num_epochs,
                                        eta_min=args.init_lr/1000)

    start_epoch = 1
    if args.resume_path != '':
        checkpoint = torch.load(args.resume_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    print("#"*20, "Start Training", "#"*20)
    
    # Track best IoU for saving best model
    best_iou = 0.0
    
    for epoch in range(start_epoch, args.num_epochs+1):
        # ========== UNFREEZE BACKBONE after N epochs ==========
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs + 1:
            model.unfreeze_backbone()
            # Recreate optimizer with ALL parameters (including backbone)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.backbone_lr, weight_decay=args.weight_decay)  # Lower LR for fine-tuning
            # Recreate scheduler for remaining epochs
            remaining_epochs = args.num_epochs - epoch + 1
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=len(train_loader) * remaining_epochs,
                eta_min=args.init_lr / 1000
            )
            print(f"[Epoch {epoch}] Backbone unfrozen, LR={args.backbone_lr:.6f}")
        
        # Train and get metrics
        loss, dice, iou = train(train_loader, model, optimizer, epoch, lr_scheduler, args)
        
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log to CSV
        log_writer.writerow([
            epoch,
            f'{loss:.4f}',
            f'{dice:.4f}',
            f'{iou:.4f}',
            f'{current_lr:.6f}',
            f'{best_iou:.4f}',
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ])
        log_file.flush()
        
        # Create checkpoint with metrics
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict(),
            'metrics': {
                'loss': loss,
                'dice': dice,
                'iou': iou
            }
        }
        
        # Save last checkpoint
        last_path = save_path + 'last.pth'
        torch.save(checkpoint, last_path)
        print(f'[Saved Last Checkpoint] {last_path}')
        
        # Save best checkpoint if IoU improved
        if iou > best_iou:
            best_iou = iou
            best_path = save_path + 'best.pth'
            torch.save(checkpoint, best_path)
            print(f'✨ [Saved Best Model] IoU: {iou:.4f} -> {best_path}')
    
    # Close log file
    log_file.close()
    print(f'[Training Complete] Log saved to: {log_path}')
