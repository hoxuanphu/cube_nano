import os
import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.cloud_dataset import CloudDataset
from models.mobilenetv3 import get_cloud_model


def set_seed(seed):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def calculate_metrics(preds, labels, threshold=0.5):
    """Calculate binary classification metrics from logits.

    Args:
        preds: (B,) logits (pre-sigmoid).
        labels: (B,) ground truth, 0 or 1.
        threshold: probability threshold for positive class.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")

    if not isinstance(preds, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("preds and labels must be torch.Tensor instances")

    # Support both (B,) and model outputs shaped (B, 1), while preventing
    # accidental broadcasting between tensors with incompatible sizes.
    preds = preds.reshape(-1)
    labels = labels.reshape(-1)
    if preds.numel() != labels.numel():
        raise ValueError(
            f"preds and labels must contain the same number of elements, "
            f"got {preds.numel()} and {labels.numel()}"
        )

    predicted_classes = torch.sigmoid(preds) >= threshold
    positive_labels = labels == 1
    negative_labels = labels == 0

    tp = (predicted_classes & positive_labels).sum().item()
    tn = (~predicted_classes & negative_labels).sum().item()
    fp = (predicted_classes & negative_labels).sum().item()
    fn = (~predicted_classes & positive_labels).sum().item()

    def safe_divide(numerator, denominator):
        return numerator / denominator if denominator else 0.0

    accuracy = safe_divide(tp + tn, tp + tn + fp + fn)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    return accuracy, precision, recall, f1

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None, threshold=0.5):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    use_amp = scaler is not None and scaler.is_enabled()
    
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            outputs = model(inputs).squeeze(1) # output shape (B,)
            loss = criterion(outputs, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        
        all_preds.append(outputs.detach().cpu())
        all_labels.append(labels.detach().cpu())
        
    avg_loss = total_loss / len(dataloader.dataset)
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    acc, prec, rec, f1 = calculate_metrics(all_preds, all_labels, threshold)
    
    return avg_loss, acc, prec, rec, f1

def val_epoch(model, dataloader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            
            all_preds.append(outputs.cpu())
            all_labels.append(labels.cpu())
            
    avg_loss = total_loss / len(dataloader.dataset)
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    acc, prec, rec, f1 = calculate_metrics(all_preds, all_labels, threshold)
    
    return avg_loss, acc, prec, rec, f1


def initialize_wandb(args, training_config):
    """Create an optional Weights & Biases run without making it a hard dependency."""
    if not args.wandb:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging was requested but the 'wandb' package is not installed. "
            "Install it with: pip install wandb"
        ) from exc

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        group=args.wandb_group,
        tags=args.wandb_tags,
        config=training_config,
        dir=args.out_dir,
        mode=args.wandb_mode,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='data/processed/train', help='Train data dir')
    parser.add_argument('--val_dir', type=str, default='data/processed/val', help='Validation data dir')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--out_dir', type=str, default='checkpoints', help='Directory to save models')
    parser.add_argument('--channels', type=int, default=4, choices=[3, 4], help='Number of input channels')
    parser.add_argument('--crop_size', type=int, default=256, help='Random crop size used for training')
    parser.add_argument(
        '--cloud_ratio_threshold', type=float, default=0.10,
        help='Minimum cloud-pixel ratio used to derive labels from crop masks',
    )
    parser.add_argument('--channel_dropout_p', type=float, default=0.3, help='Probability of zeroing the 4th channel during training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--use_pos_weight', action='store_true', default=True, help='Use pos_weight in BCEWithLogitsLoss to handle class imbalance')
    parser.add_argument('--no_pos_weight', dest='use_pos_weight', action='store_false', help='Disable pos_weight')
    parser.add_argument(
        '--pos_weight_samples_per_patch', type=int, default=8,
        help='Deterministic random crops per source patch used to estimate pos_weight',
    )
    parser.add_argument('--amp', action='store_true', default=True, help='Use Automatic Mixed Precision (AMP) training')
    parser.add_argument('--no_amp', dest='amp', action='store_false', help='Disable AMP')
    parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold for classification metrics')
    parser.add_argument('--wandb', action='store_true', help='Log training metrics to Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='cube-nano', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Optional W&B team/entity')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Optional W&B run name')
    parser.add_argument('--wandb_group', type=str, default=None, help='Optional W&B run group')
    parser.add_argument('--wandb_tags', nargs='*', default=None, help='Optional W&B run tags')
    parser.add_argument(
        '--wandb_mode', choices=('online', 'offline', 'disabled'), default='online',
        help='W&B logging mode',
    )
    args = parser.parse_args()

    # Reproducibility
    set_seed(args.seed)
    
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Datasets and Dataloaders
    print("Loading datasets...")
    # Note: If you don't have val_dir yet, you can split train dataset later.
    train_dataset = CloudDataset(
        args.train_dir,
        is_train=True,
        target_channels=args.channels,
        channel_dropout_p=args.channel_dropout_p,
        crop_size=args.crop_size,
        cloud_ratio_threshold=args.cloud_ratio_threshold,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    val_loader = None
    if os.path.exists(args.val_dir):
        val_dataset = CloudDataset(
            args.val_dir,
            is_train=False,
            target_channels=args.channels,
            crop_size=args.crop_size,
            cloud_ratio_threshold=args.cloud_ratio_threshold,
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    else:
        print(f"Validation dir {args.val_dir} not found. Training without validation.")
        
    # Model
    model = get_cloud_model(pretrained=True, num_channels=args.channels).to(device)
    
    # Loss and Optimizer
    pos_weight = None
    pos_weight_value = None
    estimated_cloud_crops = None
    estimated_clear_crops = None
    if args.use_pos_weight:
        estimated_cloud_crops, estimated_clear_crops = train_dataset.estimate_label_counts(
            samples_per_patch=args.pos_weight_samples_per_patch,
            seed=args.seed,
        )
        if estimated_cloud_crops == 0 or estimated_clear_crops == 0:
            raise ValueError(
                "Cannot compute pos_weight because the estimated crop labels do not contain "
                f"both classes (cloud={estimated_cloud_crops}, clear={estimated_clear_crops})"
            )
        pos_weight_value = estimated_clear_crops / estimated_cloud_crops
        pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
        print(
            f"Using pos_weight: {pos_weight_value:.4f} "
            f"(estimated_clear_crops={estimated_clear_crops}, "
            f"estimated_cloud_crops={estimated_cloud_crops}, "
            f"samples_per_patch={args.pos_weight_samples_per_patch})"
        )

    training_config = {
        'dataset_name': '95-Cloud',
        'train_dir': args.train_dir,
        'val_dir': args.val_dir,
        'channels': args.channels,
        'crop_size': args.crop_size,
        'cloud_ratio_threshold': args.cloud_ratio_threshold,
        'probability_threshold': args.threshold,
        'seed': args.seed,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'channel_dropout_p': args.channel_dropout_p,
        'use_pos_weight': args.use_pos_weight,
        'pos_weight_samples_per_patch': args.pos_weight_samples_per_patch,
        'estimated_cloud_crops': estimated_cloud_crops,
        'estimated_clear_crops': estimated_clear_crops,
        'pos_weight': pos_weight_value,
        'amp': args.amp and device.type == 'cuda',
        'wandb_enabled': args.wandb,
        'wandb_project': args.wandb_project if args.wandb else None,
        'wandb_entity': args.wandb_entity if args.wandb else None,
        'wandb_group': args.wandb_group if args.wandb else None,
        'wandb_tags': args.wandb_tags if args.wandb else None,
        'wandb_mode': args.wandb_mode if args.wandb else None,
    }
    config_path = os.path.join(args.out_dir, 'training_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(training_config, f, indent=2)
    print(f"Training config saved: {config_path}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # AMP scaler
    use_amp = args.amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("AMP (Automatic Mixed Precision) enabled.")
    
    best_f1 = -1.0
    best_model_path = os.path.join(args.out_dir, 'best_model.pth')
    last_model_path = os.path.join(args.out_dir, 'last_model.pth')
    wandb_run = initialize_wandb(args, training_config)
    if wandb_run is not None:
        wandb_metadata = {
            'id': getattr(wandb_run, 'id', None),
            'name': getattr(wandb_run, 'name', None),
            'project': getattr(wandb_run, 'project', args.wandb_project),
            'entity': getattr(wandb_run, 'entity', args.wandb_entity),
            'url': getattr(wandb_run, 'url', None),
            'mode': args.wandb_mode,
        }
        wandb_metadata_path = os.path.join(args.out_dir, 'wandb_run.json')
        with open(wandb_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(wandb_metadata, f, indent=2)
        print(f"W&B run: {wandb_metadata['url'] or wandb_metadata['id']}")

    try:
        for epoch in range(args.epochs):
            epoch_number = epoch + 1
            learning_rate = optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch_number}/{args.epochs}")
            train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(
                model, train_loader, criterion, optimizer, device, scaler=scaler, threshold=args.threshold
            )
            print(f"Train - Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Prec: {train_prec:.4f} | Rec: {train_rec:.4f} | F1: {train_f1:.4f}")
            torch.save(model.state_dict(), last_model_path)

            epoch_metrics = {
                'epoch': epoch_number,
                'learning_rate': learning_rate,
                'train/loss': train_loss,
                'train/accuracy': train_acc,
                'train/precision': train_prec,
                'train/recall': train_rec,
                'train/f1': train_f1,
            }

            if val_loader:
                val_loss, val_acc, val_prec, val_rec, val_f1 = val_epoch(
                    model, val_loader, criterion, device, threshold=args.threshold
                )
                print(f"Val - Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f}")

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    torch.save(model.state_dict(), best_model_path)
                    print("Saved new best model!")

                epoch_metrics.update({
                    'val/loss': val_loss,
                    'val/accuracy': val_acc,
                    'val/precision': val_prec,
                    'val/recall': val_rec,
                    'val/f1': val_f1,
                    'val/best_f1': best_f1,
                })
            else:
                torch.save(model.state_dict(), best_model_path)
                torch.save(model.state_dict(), os.path.join(args.out_dir, f'model_epoch_{epoch_number}.pth'))
                print("Saved current model as best_model.pth (no validation set available).")

            if wandb_run is not None:
                wandb_run.log(epoch_metrics, step=epoch_number)

            scheduler.step()
    finally:
        if wandb_run is not None:
            if best_f1 >= 0.0:
                wandb_run.summary['best_val_f1'] = best_f1
            wandb_run.finish()

if __name__ == '__main__':
    main()
