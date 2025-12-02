import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from datetime import datetime

from models.colonmamba import ColonMamba
from dataset import get_dataloaders, get_test_dataloader
from losses import DeepSupervisionLoss, FocalIoULoss
from metrics import MetricsCalculator


class WarmupCosineScheduler:
    """
    Learning rate scheduler with warmup followed by cosine annealing.
    
    Matches ColonFormer training strategy:
    - Warmup: Linear from 0 to base_lr over warmup_epochs
    - After: Cosine annealing to min_lr over remaining epochs
    
    Args:
        optimizer: PyTorch optimizer
        warmup_epochs (int): Number of warmup epochs (default: 1)
        total_epochs (int): Total training epochs
        base_lr (float): Base learning rate to warmup to
        min_lr (float): Minimum learning rate for cosine annealing (default: 0)
    """
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        
        # Cosine scheduler for after warmup
        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=min_lr
        )
        
        self.current_epoch = 0
    
    def step(self, epoch=None):
        """
        Update learning rate.
        
        Args:
            epoch (int, optional): Current epoch number. If None, increment internally.
        """
        if epoch is not None:
            self.current_epoch = epoch
        
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase: Linear from 0 to base_lr
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # Cosine annealing phase
            self.cosine_scheduler.step()
        
        self.current_epoch += 1
    
    def get_last_lr(self):
        """Get last learning rate (for logging)"""
        if self.current_epoch < self.warmup_epochs:
            return [self.base_lr * (self.current_epoch + 1) / self.warmup_epochs]
        else:
            return self.cosine_scheduler.get_last_lr()
    
    def state_dict(self):
        """Return state dict for checkpointing"""
        return {
            'current_epoch': self.current_epoch,
            'cosine_state': self.cosine_scheduler.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint"""
        self.current_epoch = state_dict['current_epoch']
        self.cosine_scheduler.load_state_dict(state_dict['cosine_state'])


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, writer, args):
    """
    Train for one epoch with optional multi-scale training.
    
    Args:
        args: Should have attributes:
            - multi_scale_training (bool): Enable multi-scale training
            - scale_rates (list): Scale rates for multi-scale (default: [0.75, 1.0, 1.25])
            - grad_clip (float): Gradient clipping max norm (default: 0.5)
    """
    model.train()
    
    total_loss = 0
    metrics_avg = {'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0}
    metrics_calc = MetricsCalculator(threshold=0.5)
    
    # Multi-scale configuration
    if hasattr(args, 'multi_scale_training') and args.multi_scale_training:
        size_rates = args.scale_rates if hasattr(args, 'scale_rates') else [0.75, 1.0, 1.25]
    else:
        size_rates = [1.0]  # Single scale
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        # Ensure masks have channel dimension [B, 1, H, W]
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        
        # Convert masks to float [already in 0-1 range from dataset!]
        # Dataset already converts masks to {0, 1} via binarization
        # DO NOT divide by 255 again!
        masks = masks.float()
        
        B, C, H_orig, W_orig = images.shape
        
        # ===== Multi-scale Training Loop =====
        # Train at multiple scales per batch (like ColonFormer)
        for rate_idx, rate in enumerate(size_rates):
            # Zero gradients BEFORE each scale
            optimizer.zero_grad()
            
            # Compute rescaled size (round to multiple of 32)
            if rate != 1.0:
                trainsize = int(round(H_orig * rate / 32) * 32)
                
                # Resize images and masks
                images_scaled = F.interpolate(
                    images, 
                    size=(trainsize, trainsize), 
                    mode='bilinear', 
                    align_corners=True
                )
                masks_scaled = F.interpolate(
                    masks, 
                    size=(trainsize, trainsize), 
                    mode='nearest'  # Nearest for masks to preserve binary values
                )
            else:
                images_scaled = images
                masks_scaled = masks
            
            # Forward pass with mixed precision
            with autocast(enabled=(scaler is not None)):
                outputs_tuple = model(images_scaled)

                # Convert to dict for criterion
                outputs = {
                    'main': outputs_tuple[0],      # lateral_map_5
                    'aux_1_16': outputs_tuple[1],  # lateral_map_3  
                    'aux_1_32': outputs_tuple[2],  # lateral_map_2
                    'aux_1_8': outputs_tuple[3],   # lateral_map_1
                }   
                
                loss_dict = criterion(outputs, masks_scaled)
                loss = loss_dict['total']
            
            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
                # Gradient clipping (unscale first)
                if hasattr(args, 'grad_clip') and args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                # Gradient clipping
                if hasattr(args, 'grad_clip') and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()
            
            # Calculate metrics only for scale=1.0 to avoid redundancy
            if rate == 1.0:
                with torch.no_grad():
                    pred_main = torch.sigmoid(outputs['main'])
                    batch_metrics = metrics_calc(pred_main, masks)
                    
                    for key in metrics_avg:
                        metrics_avg[key] += batch_metrics[key]
                
                # Accumulate loss (only from scale=1.0)
                total_loss += loss.item()
        
        # Update progress bar (after all scales)
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'iou': f'{batch_metrics["iou"]:.4f}',
            'dice': f'{batch_metrics["dice"]:.4f}',
            'scales': len(size_rates)
        })
        
        # Log to tensorboard every 10 batches
        if batch_idx % 10 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/Loss_total', loss.item(), global_step)
            writer.add_scalar('Train/Loss_main', loss_dict['main'].item(), global_step)
            writer.add_scalar('Train/IoU', batch_metrics['iou'], global_step)
            writer.add_scalar('Train/Dice', batch_metrics['dice'], global_step)
    
    # Average metrics
    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    for key in metrics_avg:
        metrics_avg[key] /= num_batches
    
    return avg_loss, metrics_avg


def validate(model, val_loader, criterion, device, epoch, writer):
    """Validate model"""
    model.eval()
    
    total_loss = 0
    metrics_avg = {'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0}
    metrics_calc = MetricsCalculator(threshold=0.5)
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Ensure masks have channel dimension [B, 1, H, W]
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            # Masks already in [0, 1] from dataset - just convert to float
            masks = masks.float()
            
            # Forward pass
            outputs_tuple = model(images)
            
            # Convert to dict for criterion (same as training)
            outputs = {
                'main': outputs_tuple[0],      # lateral_map_5
                'aux_1_16': outputs_tuple[1],  # lateral_map_3  
                'aux_1_32': outputs_tuple[2],  # lateral_map_2
                'aux_1_8': outputs_tuple[3],   # lateral_map_1
            }
            
            loss_dict = criterion(outputs, masks)
            loss = loss_dict['total']
            
            # Calculate metrics
            pred_main = torch.sigmoid(outputs['main'])
            batch_metrics = metrics_calc(pred_main, masks)
            
            for key in metrics_avg:
                metrics_avg[key] += batch_metrics[key]
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{batch_metrics["iou"]:.4f}',
                'dice': f'{batch_metrics["dice"]:.4f}'
            })
    
    # Average metrics
    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches
    for key in metrics_avg:
        metrics_avg[key] /= num_batches
    
    # Log to tensorboard
    writer.add_scalar('Val/Loss', avg_loss, epoch)
    writer.add_scalar('Val/IoU', metrics_avg['iou'], epoch)
    writer.add_scalar('Val/Dice', metrics_avg['dice'], epoch)
    writer.add_scalar('Val/Precision', metrics_avg['precision'], epoch)
    writer.add_scalar('Val/Recall', metrics_avg['recall'], epoch)
    
    return avg_loss, metrics_avg


def save_checkpoint(model, optimizer, epoch, metrics, save_path, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = save_path.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_path)
        print(f'✓ Saved best model to {best_path}')


def train(args):
    """Main training function"""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Setup tensorboard
    log_dir = os.path.join(args.save_dir, 'logs')
    writer = SummaryWriter(log_dir)
    
    # Create dataloaders
    print('\n[1/5] Loading datasets...')
    train_loader, val_loader = get_dataloaders(
        train_img_dir=os.path.join(args.train_dir, 'image'),
        train_mask_dir=os.path.join(args.train_dir, 'mask'),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        val_split=args.val_split,
    )
    
    # Create model
    print('\n[2/5] Creating model...')
    model = ColonMamba(
        num_classes=1,
        pretrained_resnet=True,  
        pretrained_vmamba=True,  
        vmamba_checkpoint_path=args.vmamba_ckpt,
        channel_mode=args.channel_mode,
    ).to(device)
    
    param_count, size_mb = model.get_model_size()
    print(f'Model parameters: {param_count:,} ({param_count/1e6:.2f}M)')
    print(f'Model size: {size_mb:.2f} MB')
    
    # Create loss function
    print('\n[3/5] Setting up training...')
    # Use ColonFormer-style Focal + IoU loss with edge-aware weighting
    from losses import FocalIoULoss
    base_loss = FocalIoULoss(
        focal_alpha=0.25,  # Class balancing weight
        focal_gamma=2.0,   # Focusing parameter
        focal_weight=0.5,  # Weight for Focal loss
        iou_weight=0.5,    # Weight for IoU loss
        use_edge_weight=args.use_edge_weight,  # Edge-aware weighting
        edge_kernel_size=args.edge_kernel_size if hasattr(args, 'edge_kernel_size') else 31,
        edge_boost=args.edge_boost if hasattr(args, 'edge_boost') else 5.0,
    )
    criterion = DeepSupervisionLoss(
        base_loss=base_loss,
        aux_weights=[args.aux_weight] * 3,  # Equal weights for 3 aux outputs
        main_weight=1.0,
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Learning rate scheduler with warmup (matches ColonFormer)
    lr_scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=1,              # Linear warmup for 1 epoch
        total_epochs=args.epochs,
        base_lr=args.lr,
        min_lr=0
    )
    print(f'Scheduler: WarmupCosineScheduler (warmup=1 epoch, then cosine to 0)')
    
    # Mixed precision scaler
    scaler = GradScaler() if args.mixed_precision else None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_iou = 0.0
    
    if args.resume:
        if os.path.exists(args.resume):
            print(f'Resuming from checkpoint: {args.resume}')
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_iou = checkpoint['metrics'].get('iou', 0.0)
            print(f'Resumed from epoch {start_epoch}, best IoU: {best_iou:.4f}')
        else:
            print(f'Warning: Checkpoint {args.resume} not found, starting from scratch')
    
    # Training loop
    print(f'\n[4/5] Starting training for {args.epochs} epochs...')
    print('='*60)
    
    for epoch in range(start_epoch, args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        print('-' * 40)
        
        # Train
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, writer, args
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, epoch, writer
        )
        
        # Update learning rate
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Train/LR', current_lr, epoch)
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        print(f'  Train IoU: {train_metrics["iou"]:.4f} | Val IoU: {val_metrics["iou"]:.4f}')
        print(f'  Train Dice: {train_metrics["dice"]:.4f} | Val Dice: {val_metrics["dice"]:.4f}')
        print(f'  LR: {current_lr:.6f}')
        
        # Save checkpoint
        is_best = val_metrics['iou'] > best_iou
        if is_best:
            best_iou = val_metrics['iou']
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch{epoch+1}.pth')
        save_checkpoint(model, optimizer, epoch, val_metrics, checkpoint_path, is_best=is_best)
        
        # Save every N epochs
        if (epoch + 1) % args.save_freq == 0:
            print(f'✓ Saved checkpoint to {checkpoint_path}')
    
    print('\n' + '='*60)
    print(f'[5/5] Training completed!')
    print(f'Best validation IoU: {best_iou:.4f}')
    print(f'Checkpoints saved to: {args.save_dir}')
    
    writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train ColonMamba for polyp segmentation')
    
    # Data paths
    parser.add_argument('--train_dir', type=str, default='data/TrainDataset',
                       help='Training dataset directory (contains image/ and mask/ subdirs)')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio (default: 0.2)')
    
    # Model settings
    parser.add_argument('--vmamba_ckpt', type=str, 
                       default='pretrained/vssmtiny_dp01_ckpt_epoch_292.pth',
                       help='Path to VMamba pretrained')
    parser.add_argument('--channel_mode', type=str, default='project',
                       choices=['project', 'adapt'],
                       help='Channel adaptation mode: '
                        'project (default) = [320,512] fair comparison | '
                        'adapt = [384,768] best performance')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--aux_weight', type=float, default=0.4,
                       help='Auxiliary loss weight')
    parser.add_argument('--image_size', type=int, default=352,
                       help='Input image size')
    
    # Loss configuration
    parser.add_argument('--use_edge_weight', action='store_true', default=True,
                       help='Use edge-aware weighting in loss')
    parser.add_argument('--edge_kernel_size', type=int, default=31,
                       help='Kernel size for edge-aware weighting')
    parser.add_argument('--edge_boost', type=float, default=5.0,
                       help='Edge weight boost factor')
    
    # Multi-scale training 
    parser.add_argument('--multi_scale_training', action='store_true', default=True,
                       help='Enable multi-scale training (3 scales per batch)')
    parser.add_argument('--scale_rates', type=float, nargs='+', default=[0.75, 1.0, 1.25],
                       help='Scale rates for multi-scale training')
    
    # Gradient clipping
    parser.add_argument('--grad_clip', type=float, default=0.5,
                       help='Gradient clipping max norm (0 to disable)')
    
    # System settings
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')
    parser.add_argument('--mixed_precision', action='store_true', default=False,
                       help='Use mixed precision training')
    
    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--save_freq', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()

    # Print channel mode
    print('\n' + '='*60)
    print(f'Channel Mode: {args.channel_mode.upper()}')
    if args.channel_mode == 'project':
        print('  Strategy: Fair comparison with ColonFormer')
        print('  Encoder outputs: [64, 128, 320, 512]')
    else:
        print('  Strategy: Best performance (adapt to VMamba)')
        print('  Encoder outputs: [64, 128, 384, 768]')
    print('='*60)
    
    # Print configuration
    print('='*60)
    print('ColonMamba Training Configuration')
    print('='*60)
    for arg, value in vars(args).items():
        print(f'{arg:20s}: {value}')
    print('='*60)
    
    # Start training
    train(args)


if __name__ == '__main__':
    main()
