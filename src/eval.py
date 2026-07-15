import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from data.cloud_dataset import CloudDataset
from models.mobilenetv3 import get_cloud_model
from train import calculate_metrics, set_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, default='data/processed/test', help='Test data dir')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth', help='Path to trained model')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--channels', type=int, default=4, choices=[3, 4], help='Number of input channels')
    parser.add_argument('--crop_size', type=int, default=256, help='Evaluation crop/window size')
    parser.add_argument(
        '--cloud_ratio_threshold', type=float, default=0.10,
        help='Minimum cloud-pixel ratio used to derive labels from crop masks',
    )
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed recorded for reproducibility')
    parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold for classification')
    args = parser.parse_args()

    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists(args.test_dir):
        print(f"Error: Test directory {args.test_dir} not found.")
        return
        
    print("Loading test dataset...")
    test_dataset = CloudDataset(
        args.test_dir,
        is_train=False,
        target_channels=args.channels,
        crop_size=args.crop_size,
        cloud_ratio_threshold=args.cloud_ratio_threshold,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    print(f"Loading model from {args.model_path}...")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    model = get_cloud_model(pretrained=False, num_channels=args.channels)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    criterion = nn.BCEWithLogitsLoss()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            
            all_preds.append(outputs.cpu())
            all_labels.append(labels.cpu())
            
    avg_loss = total_loss / len(test_loader.dataset)
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    acc, prec, rec, f1 = calculate_metrics(all_preds, all_labels, threshold=args.threshold)
    
    results = {
        'config': {
            'dataset_name': '95-Cloud',
            'model_path': args.model_path,
            'test_dir': args.test_dir,
            'channels': args.channels,
            'batch_size': args.batch_size,
            'crop_size': args.crop_size,
            'cloud_ratio_threshold': args.cloud_ratio_threshold,
            'probability_threshold': args.threshold,
            'seed': args.seed
        },
        'metrics': {
            'Loss': round(avg_loss, 4),
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4)
        }
    }
    
    print("\n--- Evaluation Results ---")
    for k, v in results['metrics'].items():
        print(f"{k}: {v}")
        
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/eval_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print("\nResults saved to results/eval_metrics.json")

if __name__ == '__main__':
    main()
