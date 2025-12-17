import argparse
import os
import csv
from datetime import datetime
import numpy as np
import cv2
from tqdm import tqdm
from glob import glob
import torch
import torch.nn.functional as F

from mmseg import __version__
from mmseg.models.segmentors import ColonFormer as UNet

from train import get_backbone_cfg

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
        original_size = mask.shape  # Store original size for resizing prediction

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

        return np.asarray(image), np.asarray(mask), img_path, original_size

epsilon = 1e-7

def recall_np(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + epsilon)
    return recall

def precision_np(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + epsilon)
    return precision

def dice_np(y_true, y_pred):
    precision = precision_np(y_true, y_pred)
    recall = recall_np(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+epsilon))

def iou_np(y_true, y_pred):
    intersection = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    union = np.sum(y_true)+np.sum(y_pred)-intersection
    return intersection/(union+epsilon)

def get_scores(gts, prs):
    if len(gts) == 0:
        print("Warning: No test images found!")
        return (0, 0, 0, 0)
    
    mean_precision = 0
    mean_recall = 0
    mean_iou = 0
    mean_dice = 0
    for gt, pr in zip(gts, prs):
        mean_precision += precision_np(gt, pr)
        mean_recall += recall_np(gt, pr)
        mean_iou += iou_np(gt, pr)
        mean_dice += dice_np(gt, pr)

    mean_precision /= len(gts)
    mean_recall /= len(gts)
    mean_iou /= len(gts)
    mean_dice /= len(gts)        
    
    print("scores: dice={:.4f}, miou={:.4f}, precision={:.4f}, recall={:.4f}".format(
        mean_dice, mean_iou, mean_precision, mean_recall))

    return (mean_iou, mean_dice, mean_precision, mean_recall)


def get_config_name(args):
    """Generate config name based on arguments"""
    parts = [args.backbone]
    if args.attention_type == 'ss2d':
        parts.append("ss2d")
    if args.use_local_global:
        parts.append("local_global")
    return "_".join(parts)


def save_metrics_to_csv(metrics, args, output_dir):
    """Save metrics to CSV file"""
    config_name = get_config_name(args)
    csv_path = os.path.join(output_dir, f"metrics_{config_name}.csv")
    
    mean_iou, mean_dice, mean_precision, mean_recall = metrics
    
    # Check if file exists to determine if we need to write header
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'backbone', 'attention_type', 'use_local_global', 'weight', 'test_path', 
                           'dice', 'iou', 'precision', 'recall'])
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            args.backbone,
            args.attention_type,
            args.use_local_global,
            args.weight,
            args.test_path,
            f'{mean_dice:.4f}',
            f'{mean_iou:.4f}',
            f'{mean_precision:.4f}',
            f'{mean_recall:.4f}'
        ])
    
    print(f"[Saved Metrics] {csv_path}")
    return csv_path


def inference_single_dataset(model, dataset_path, dataset_name, config_name, args):
    """Run inference on a single dataset"""
    print(f"\n{'#'*20} Testing: {dataset_name} {'#'*20}")
    model.eval()
    
    # Setup output directory for masks if enabled
    mask_output_dir = None
    if args.save_masks:
        mask_output_dir = os.path.join(args.output_dir, f"masks_{config_name}", dataset_name)
        os.makedirs(mask_output_dir, exist_ok=True)
    
    X_test = glob('{}/images/*'.format(dataset_path))
    X_test.sort()
    y_test = glob('{}/masks/*'.format(dataset_path))
    y_test.sort()
    
    if len(X_test) == 0:
        print(f"Warning: No images found in {dataset_path}/images/")
        return None

    test_dataset = Dataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    gts = []
    prs = []
    
    for i, pack in enumerate(tqdm(test_loader, desc=f"Testing {dataset_name}"), start=1):
        image, gt, img_path, original_size = pack
        gt = gt[0][0]
        gt = np.asarray(gt, np.float32)
        image = image.cuda()

        with torch.no_grad():
            res, res2, res3, res4 = model(image)
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        pr = res.round()
        gts.append(gt)
        prs.append(pr)
        
        # Save predicted mask if enabled
        if args.save_masks and mask_output_dir:
            img_name = os.path.basename(img_path[0])
            base_name = os.path.splitext(img_name)[0]
            
            orig_h = original_size[0].item() if torch.is_tensor(original_size[0]) else int(original_size[0])
            orig_w = original_size[1].item() if torch.is_tensor(original_size[1]) else int(original_size[1])
            if pr.shape != (orig_h, orig_w):
                pr_resized = cv2.resize(pr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            else:
                pr_resized = pr
            
            mask_save_path = os.path.join(mask_output_dir, f"{base_name}_pred.png")
            cv2.imwrite(mask_save_path, (pr_resized * 255).astype(np.uint8))
    
    # Calculate metrics
    metrics = get_scores(gts, prs)
    
    if args.save_masks and mask_output_dir:
        print(f"[Saved Masks] {len(prs)} masks -> {mask_output_dir}")
    
    return {
        'dataset': dataset_name,
        'num_images': len(gts),
        'metrics': metrics
    }


def inference(model, args):
    """Run inference on all datasets in test_path"""
    config_name = get_config_name(args)
    test_path = args.test_path
    
    # Check if test_path contains subdatasets (folders with images/masks inside)
    subdatasets = []
    for item in sorted(os.listdir(test_path)):
        item_path = os.path.join(test_path, item)
        if os.path.isdir(item_path):
            # Check if this folder has images/ subfolder
            if os.path.exists(os.path.join(item_path, 'images')):
                subdatasets.append((item, item_path))
    
    # If no subdatasets found, treat test_path as single dataset
    if len(subdatasets) == 0:
        if os.path.exists(os.path.join(test_path, 'images')):
            subdatasets = [('test', test_path)]
        else:
            print(f"Error: No images found in {test_path}")
            return
    
    print(f"\n[Found {len(subdatasets)} dataset(s)]: {[d[0] for d in subdatasets]}")
    
    # Run inference on each dataset
    all_results = []
    for dataset_name, dataset_path in subdatasets:
        result = inference_single_dataset(model, dataset_path, dataset_name, config_name, args)
        if result:
            all_results.append(result)
    
    # Save combined results to CSV
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"metrics_{config_name}.csv")
    
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'backbone', 'attention_type', 'use_local_global', 'weight', 'dataset', 
                           'num_images', 'dice', 'iou', 'precision', 'recall'])
        for result in all_results:
            mean_iou, mean_dice, mean_precision, mean_recall = result['metrics']
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                args.backbone,
                args.attention_type,
                args.use_local_global,
                args.weight,
                result['dataset'],
                result['num_images'],
                f'{mean_dice:.4f}',
                f'{mean_iou:.4f}',
                f'{mean_precision:.4f}',
                f'{mean_recall:.4f}'
            ])
    
    print(f"\n{'='*50}")
    print(f"[Results saved to] {csv_path}")
    print(f"[Tested {len(all_results)} datasets]")
    
    # Print summary table
    print(f"\n{'Dataset':<20} {'Images':<10} {'Dice':<10} {'mIoU':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 70)
    for r in all_results:
        iou, dice, prec, rec = r['metrics']
        print(f"{r['dataset']:<20} {r['num_images']:<10} {dice:<10.4f} {iou:<10.4f} {prec:<10.4f} {rec:<10.4f}")
    
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str,
                        default='b3', help='Backbone: b0-b5, vmamba_tiny, vmamba_small, vmamba_base')
    parser.add_argument('--weight', type=str, required=True,
                        help='Path to model checkpoint (required)')
    parser.add_argument('--test_path', type=str,
                        default='./data/TestDataset', help='Path to test dataset')
    parser.add_argument('--output_dir', type=str,
                        default='./results', help='Directory to save results (CSV and masks)')
    parser.add_argument('--save_masks', action='store_true',
                        help='Save predicted masks to output directory')
    parser.add_argument('--attention_type', type=str, default='ss2d',
                        choices=['aa_kernel', 'ss2d'],
                        help='Type of spatial attention: aa_kernel or ss2d (default: ss2d)')
    parser.add_argument('--use_local_global', action='store_true',
                        help='Enable 2-Branch Bottleneck (Local DW-Conv + Global Attention)')
    args = parser.parse_args()

    backbone_cfg, in_channels = get_backbone_cfg(args.backbone)
    
    # Determine pretrained path (same as original code)
    if 'vmamba' in args.backbone:
        pretrained_path = 'pretrained/vssm_{}.pth'.format(args.backbone.replace('vmamba_', ''))
    else:
        pretrained_path = 'pretrained/mit_{}.pth'.format(args.backbone)
    
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
                attention_type=args.attention_type,
                use_local_global=args.use_local_global,
                pretrained=pretrained_path).cuda()

    # Load checkpoint (overwrites pretrained backbone weights)
    checkpoint = torch.load(args.weight, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(checkpoint['state_dict'])
    print(f"[Loaded] {args.weight}")

    inference(model, args)



