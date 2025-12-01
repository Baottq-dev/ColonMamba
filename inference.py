"""
Inference Script for ColonMamba

Supports:
- Single image inference
- Batch inference on test datasets
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


def load_model(checkpoint_path, device='cuda'):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        device (str): Device to load model on
    
    Returns:
        nn.Module: Loaded model in eval mode
    """
    print(f'Loading model from {checkpoint_path}...')
    
    # Create model
    model = ColonMamba(
        num_classes=1,
        pretrained_resnet=False,  # Not needed for inference
        pretrained_vmamba=False,
        use_mrr=True,
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
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
        pred = model.inference(image_tensor)  # Returns probabilities [0, 1]
    
    # Convert to numpy and resize to original size
    pred_mask = pred.squeeze().cpu().numpy()
    pred_mask = cv2.resize(pred_mask, (original_size[1], original_size[0]))
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    
    return pred_mask, image


def evaluate_on_dataset(model, test_loader, device='cuda', save_dir=None):
    """
    Evaluate model on a test dataset.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test set
        device (str): Device
        save_dir (str): Optional directory to save predictions
    
    Returns:
        dict: Average metrics
    """
    model.eval()
    
    metrics_calc = MetricsCalculator(threshold=0.5)
    all_metrics = []
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        pred_dir = os.path.join(save_dir, 'predictions')
        overlay_dir = os.path.join(save_dir, 'overlays')
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(overlay_dir, exist_ok=True)
    
    pbar = tqdm(test_loader, desc='Evaluating')
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            names = batch['name']
            
            # Predict
            pred = model.inference(images)
            
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
                    # Get prediction and denormalize image
                    pred_np = (pred[i].squeeze().cpu().numpy() * 255).astype(np.uint8)
                    
                    # Save prediction
                    pred_path = os.path.join(pred_dir, names[i])
                    cv2.imwrite(pred_path, pred_np)
    
    # Calculate average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    return avg_metrics


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
    parser = argparse.ArgumentParser(description='ColonMamba Inference')
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--image_size', type=int, default=352,
                       help='Input image size')
    
    # Inference mode
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'dataset'],
                       help='Inference mode: single image or full dataset')
    
    # Single image mode
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single input image')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory for predictions')
    
    # Dataset mode
    parser.add_argument('--test_img_dir', type=str, default=None,
                       help='Test images directory')
    parser.add_argument('--test_mask_dir', type=str, default=None,
                       help='Test masks directory')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for dataset inference')
    
    # System
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    model = load_model(args.checkpoint, device)
    
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
        
        # Visualize if requested
        if args.visualize:
            # Try to load ground truth if exists
            gt_path = args.image.replace('image', 'mask').replace('images', 'masks')
            if os.path.exists(gt_path):
                gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                vis_path = os.path.join(args.output, f'{base_name}_vis.png')
                visualize_prediction(original_image, gt_mask, pred_mask, vis_path)
    
    elif args.mode == 'dataset':
        # Dataset inference
        if args.test_img_dir is None or args.test_mask_dir is None:
            raise ValueError('--test_img_dir and --test_mask_dir must be specified for dataset mode')
        
        print(f'\nEvaluating on dataset: {args.test_img_dir}')
        
        # Create test dataloader
        test_loader = get_test_dataloader(
            test_img_dir=args.test_img_dir,
            test_mask_dir=args.test_mask_dir,
            batch_size=args.batch_size,
            num_workers=4,
            image_size=args.image_size,
        )
        
        # Evaluate
        metrics = evaluate_on_dataset(
            model, test_loader, device,
            save_dir=args.output if args.visualize else None
        )
        
        # Print results
        print('\n' + '='*50)
        print('Evaluation Results:')
        print('='*50)
        print(f'IoU:         {metrics["iou"]:.4f}')
        print(f'Dice:        {metrics["dice"]:.4f}')
        print(f'Precision:   {metrics["precision"]:.4f}')
        print(f'Recall:      {metrics["recall"]:.4f}')
        print(f'Specificity: {metrics["specificity"]:.4f}')
        print(f'MAE:         {metrics["mae"]:.4f}')
        print('='*50)
        
        # Save results to file
        results_path = os.path.join(args.output, 'results.txt')
        with open(results_path, 'w') as f:
            f.write('ColonMamba Evaluation Results\n')
            f.write('='*50 + '\n')
            for key, value in metrics.items():
                f.write(f'{key:12s}: {value:.4f}\n')
        
        print(f'\n✓ Results saved to {results_path}')


if __name__ == '__main__':
    main()
