"""
Inference Script for ColonMamba

Supports:
- Single image inference
- Batch inference on multiple test datasets (CVC-300, CVC-ClinicDB, CVC-ColonDB, ETIS, Kvasir)
- Evaluation metrics calculation
- Visualization of predictions
"""

import os
import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

from models.colonmamba import ColonMamba
from dataset import get_test_dataloader, get_val_transform
from metrics import MetricsCalculator


# Standard test datasets
TEST_DATASETS = {
    'CVC-300': 'data/TestDataset/CVC-300',
    'CVC-ClinicDB': 'data/TestDataset/CVC-ClinicDB',
    'CVC-ColonDB': 'data/TestDataset/CVC-ColonDB',
    'ETIS': 'data/TestDataset/ETIS-LaribPolypDB',
    'Kvasir': 'data/TestDataset/Kvasir',
}


def load_model(checkpoint_path, device='cuda', channel_mode='project'):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        device (str): Device to load model on
        channel_mode (str): 'project' or 'adapt'
    
    Returns:
        nn.Module: Loaded model in eval mode
    """
    print(f'Loading model from {checkpoint_path}...')
    
    # Create model
    model = ColonMamba(
        num_classes=1,
        pretrained_resnet=False,  # Not needed for inference
        pretrained_vmamba=False,
        channel_mode=channel_mode,
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'epoch' in checkpoint:
            print(f'  Checkpoint from epoch {checkpoint["epoch"] + 1}')
        if 'metrics' in checkpoint:
            print(f'  Val IoU: {checkpoint["metrics"].get("iou", 0):.4f}')
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print('✓ Model loaded successfully')
    
    return model


def predict_single_image(model, image_path, device='cuda', image_size=352):
    """
    Predict on a single image.
    
    Args:
        model: Trained model
        image_path (str): Path to input image
        device (str): Device
        image_size (int): Image size
    
    Returns:
        tuple: (prediction_mask, original_image)
    """
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]
    
    # Apply transform
    transform = get_val_transform(image_size)
    augmented = transform(image=image)
    image_tensor = augmented['image'].unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)  # Returns tuple of 4 outputs
        pred = torch.sigmoid(outputs[0])  # Main prediction
    
    # Convert to numpy and resize to original size
    pred_mask = pred.squeeze().cpu().numpy()
    pred_mask = cv2.resize(pred_mask, (original_size[1], original_size[0]))
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    
    return pred_mask, image


def evaluate_on_dataset(model, dataset_name, dataset_path, device='cuda', 
                        image_size=352, batch_size=1, save_dir=None):
    """
    Evaluate model on a single test dataset.
    
    Args:
        model: Trained model
        dataset_name (str): Name of dataset
        dataset_path (str): Path to dataset root
        device (str): Device
        image_size (int): Image size
        batch_size (int): Batch size
        save_dir (str): Optional directory to save predictions
    
    Returns:
        dict: Average metrics
    """
    print(f'\n{"="*60}')
    print(f'Evaluating on: {dataset_name}')
    print(f'{"="*60}')
    
    # Get image and mask directories
    img_dir = os.path.join(dataset_path, 'images')
    mask_dir = os.path.join(dataset_path, 'masks')
    
    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        print(f'⚠️  Skipping {dataset_name}: images or masks directory not found')
        return None
    
    # Count images
    num_images = len([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.bmp'))])
    print(f'Number of images: {num_images}')
    
    # Create test dataloader
    test_loader = get_test_dataloader(
        test_img_dir=img_dir,
        test_mask_dir=mask_dir,
        batch_size=batch_size,
        num_workers=4,
        image_size=image_size,
    )
    
    model.eval()
    
    metrics_calc = MetricsCalculator(threshold=0.5)
    all_metrics = []
    
    if save_dir:
        dataset_save_dir = os.path.join(save_dir, dataset_name)
        os.makedirs(dataset_save_dir, exist_ok=True)
        pred_dir = os.path.join(dataset_save_dir, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)
    
    pbar = tqdm(test_loader, desc=f'{dataset_name}')
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            names = batch['name']
            
            # Ensure masks have channel dimension and float dtype
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            masks = masks.float() / 255.0
            
            # Predict
            outputs = model(images)  # Returns tuple of 4 outputs
            pred = torch.sigmoid(outputs[0])  # Main prediction
            
            # Calculate metrics
            batch_metrics = metrics_calc(pred, masks)
            all_metrics.append(batch_metrics)
            
            # Update progress bar
            pbar.set_postfix({
                'IoU': f'{batch_metrics["iou"]:.4f}',
                'Dice': f'{batch_metrics["dice"]:.4f}'
            })
            
            # Save predictions if requested
            if save_dir:
                for i in range(len(names)):
                    # Get prediction
                    pred_np = (pred[i].squeeze().cpu().numpy() * 255).astype(np.uint8)
                    
                    # Save prediction
                    pred_path = os.path.join(pred_dir, names[i])
                    cv2.imwrite(pred_path, pred_np)
    
    # Calculate average metrics
    if not all_metrics:
        return None
    
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    # Print results
    print(f'\n{dataset_name} Results:')
    print(f'  IoU:         {avg_metrics["iou"]:.4f}')
    print(f'  Dice:        {avg_metrics["dice"]:.4f}')
    print(f'  Precision:   {avg_metrics["precision"]:.4f}')
    print(f'  Recall:      {avg_metrics["recall"]:.4f}')
    
    return avg_metrics


def evaluate_all_datasets(model, device='cuda', image_size=352, batch_size=1, 
                         save_dir=None, datasets=None):
    """
    Evaluate model on all standard test datasets.
    
    Args:
        model: Trained model
        device (str): Device
        image_size (int): Image size
        batch_size (int): Batch size
        save_dir (str): Optional directory to save predictions
        datasets (list): List of dataset names to evaluate (default: all)
    
    Returns:
        dict: Results for each dataset
    """
    if datasets is None:
        datasets = list(TEST_DATASETS.keys())
    
    all_results = {}
    
    for dataset_name in datasets:
        if dataset_name not in TEST_DATASETS:
            print(f'⚠️  Unknown dataset: {dataset_name}')
            continue
        
        dataset_path = TEST_DATASETS[dataset_name]
        
        if not os.path.exists(dataset_path):
            print(f'⚠️  Dataset path not found: {dataset_path}')
            continue
        
        results = evaluate_on_dataset(
            model, dataset_name, dataset_path,
            device=device,
            image_size=image_size,
            batch_size=batch_size,
            save_dir=save_dir
        )
        
        if results is not None:
            all_results[dataset_name] = results
    
    # Print summary
    if all_results:
        print(f'\n{"="*60}')
        print('SUMMARY - All Datasets')
        print(f'{"="*60}')
        print(f'{"Dataset":<20} {"IoU":>8} {"Dice":>8} {"Precision":>10} {"Recall":>8}')
        print('-' * 60)
        
        for dataset_name, metrics in all_results.items():
            print(f'{dataset_name:<20} {metrics["iou"]:>8.4f} {metrics["dice"]:>8.4f} '
                  f'{metrics["precision"]:>10.4f} {metrics["recall"]:>8.4f}')
        
        # Calculate mean across datasets
        mean_iou = np.mean([m["iou"] for m in all_results.values()])
        mean_dice = np.mean([m["dice"] for m in all_results.values()])
        mean_precision = np.mean([m["precision"] for m in all_results.values()])
        mean_recall = np.mean([m["recall"] for m in all_results.values()])
        
        print('-' * 60)
        print(f'{"MEAN":<20} {mean_iou:>8.4f} {mean_dice:>8.4f} '
              f'{mean_precision:>10.4f} {mean_recall:>8.4f}')
        print('=' * 60)
        
        # Save summary to file
        if save_dir:
            summary_path = os.path.join(save_dir, 'summary.txt')
            with open(summary_path, 'w') as f:
                f.write('ColonMamba Evaluation Summary\n')
                f.write('='*60 + '\n\n')
                f.write(f'{"Dataset":<20} {"IoU":>8} {"Dice":>8} {"Precision":>10} {"Recall":>8}\n')
                f.write('-' * 60 + '\n')
                
                for dataset_name, metrics in all_results.items():
                    f.write(f'{dataset_name:<20} {metrics["iou"]:>8.4f} {metrics["dice"]:>8.4f} '
                           f'{metrics["precision"]:>10.4f} {metrics["recall"]:>8.4f}\n')
                
                f.write('-' * 60 + '\n')
                f.write(f'{"MEAN":<20} {mean_iou:>8.4f} {mean_dice:>8.4f} '
                       f'{mean_precision:>10.4f} {mean_recall:>8.4f}\n')
            
            print(f'\n✓ Summary saved to {summary_path}')
    
    return all_results


def visualize_prediction(image, gt_mask, pred_mask, save_path=None):
    """
    Visualize prediction alongside ground truth.
    
    Args:
        image (ndarray): Original image [H, W, 3]
        gt_mask (ndarray): Ground truth mask [H, W]
        pred_mask (ndarray): Predicted mask [H, W]
        save_path (str): Optional path to save visualization
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # Overlay
    overlay = image.copy()
    # Green for prediction, Red for ground truth
    overlay[pred_mask > 127] = [0, 255, 0]  # Green
    overlay[gt_mask > 127] = [255, 0, 0]  # Red
    # Yellow where they overlap
    overlap = (pred_mask > 127) & (gt_mask > 127)
    overlay[overlap] = [255, 255, 0]
    
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay (GT: Red, Pred: Green)')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved visualization to {save_path}')
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='ColonMamba Inference and Evaluation')
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--image_size', type=int, default=352,
                       help='Input image size')
    parser.add_argument('--channel_mode', type=str, default='project',
                       choices=['project', 'adapt'],
                       help='Channel mode (should match training)')
    
    # Inference mode
    parser.add_argument('--mode', type=str, default='all_datasets', 
                       choices=['single', 'dataset', 'all_datasets'],
                       help='Inference mode: single image, single dataset, or all standard datasets')
    
    # Single image mode
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single input image')
    
    # Dataset mode
    parser.add_argument('--dataset', type=str, default=None,
                       choices=list(TEST_DATASETS.keys()),
                       help='Specific dataset to evaluate')
    
    # Output
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for predictions and results')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save prediction masks')
    
    # System
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for dataset inference')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    model = load_model(args.checkpoint, device, args.channel_mode)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    if args.mode == 'single':
        # Single image inference
        if args.image is None:
            raise ValueError('--image must be specified for single mode')
        
        print(f'\nPredicting on: {args.image}')
        
        pred_mask, original_image = predict_single_image(
            model, args.image, device, args.image_size
        )
        
        # Save prediction
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        pred_path = os.path.join(args.output, f'{base_name}_pred.png')
        cv2.imwrite(pred_path, pred_mask)
        print(f'✓ Saved prediction to {pred_path}')
    
    elif args.mode == 'dataset':
        # Single dataset evaluation
        if args.dataset is None:
            raise ValueError('--dataset must be specified for dataset mode')
        
        dataset_path = TEST_DATASETS[args.dataset]
        
        evaluate_on_dataset(
            model, args.dataset, dataset_path,
            device=device,
            image_size=args.image_size,
            batch_size=args.batch_size,
            save_dir=args.output if args.save_predictions else None
        )
    
    elif args.mode == 'all_datasets':
        # Evaluate on all standard datasets
        evaluate_all_datasets(
            model,
            device=device,
            image_size=args.image_size,
            batch_size=args.batch_size,
            save_dir=args.output if args.save_predictions else None
        )


if __name__ == '__main__':
    main()
