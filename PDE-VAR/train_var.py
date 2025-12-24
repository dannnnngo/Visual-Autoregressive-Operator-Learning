import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

# Import VQVAE modules
from models.vqvae import VQVAE
from models.basic_vae import Encoder, Decoder
from models.quant import VectorQuantizer2

# Assuming autoregressive_sr is available
from models.pdevar import PDEVAR

# =================================== Dataset and helper functions
class PDEDataset(Dataset):
    """
    Dataset for PDE data (D and u fields).
    Similar to ImagePredictionDataset in train.py
    """
    
    def __init__(
        self,
        D_images: torch.Tensor,  # (N, 256, 256)
        u_averages: torch.Tensor,  # (N, 16, 16)
    ):
        """
        Args:
            D_images: Full resolution D field images (N, 256, 256)
            u_averages: Averaged u field images (N, 16, 16)
        """
        self.D_images = D_images
        self.u_averages = u_averages
        
    def __len__(self):
        return len(self.D_images)
    
    def __getitem__(self, idx):
        D_img = self.D_images[idx]  # (256, 256)
        u_avg = self.u_averages[idx]  # (16, 16)
        
        # Convert to 3-channel format for VQVAE (repeat grayscale)
        # and normalize to [-1, 1]
        D_img_3ch = D_img.unsqueeze(0).repeat(3, 1, 1)  # (3, 256, 256)
        # Normalize to [-1, 1] if not already
        if D_img_3ch.min() >= 0 and D_img_3ch.max() <= 1:
            D_img_3ch = 2.0 * D_img_3ch - 1.0

        return {
            'D': D_img_3ch,  # High-resolution D field (target) [3, 256, 256]
            'u': u_avg,  # Low-resolution u field (conditioning) [16, 16]
        }


def image_to_average(image, target_size=(16, 16)):
    """
    Average image from (B, H, W) or (H, W) to target size.
    Same function as in train.py
    """
    if len(image.shape) == 2:
        image = image.unsqueeze(0)  # (1, H, W)
    
    B, H, W = image.shape
    target_H, target_W = target_size
    assert H % target_H == 0 and W % target_W == 0, "Image dimensions must be divisible by target size"
    block_H, block_W = H // target_H, W // target_W
    image = image.view(B, target_H, block_H, target_W, block_W)
    averaged = image.mean(dim=(2, 4))  # (B, target_H, target_W)
    return averaged

def compute_loss(logits_BLV, target_list):
    B, L, V = logits_BLV.shape
    
    target_BL = torch.cat(target_list, dim=1)

    # Flatten logits and targets
    logits_flat = logits_BLV.reshape(-1, V)  # (B*L) × V
    target_flat = target_BL.reshape(-1)  # (B*L)
    
    # Compute loss
    loss = F.cross_entropy(logits_flat, target_flat)
    
    with torch.no_grad():
        pred_flat = logits_flat.argmax(dim=-1)
        accuracy = (pred_flat == target_flat).float().mean()
    
    return loss, accuracy

def plot_curves(train_losses, train_accs, val_losses, val_accs, save_path):
    """Plot training and validation curves"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curve
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc')
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Curves saved to {save_path}")

# ================================ Training Process
def train_epoch(model, vqvae, dataloader, optimizer, device, epoch, patch_nums):
    """Train for one epoch"""
    model.train()
    vqvae.eval()
    
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        D_target = batch['D'].to(device)  # B×3×256×256
        u_avg = batch['u'].to(device)  # B×16×16
        
        if u_avg.dim() == 3:
            u_avg = u_avg.unsqueeze(1)  # B×1×16×16 for spatial conditioning
        
        with torch.no_grad():
            f = vqvae.quant_conv(vqvae.encoder(D_target))
            gt_tokens, gt_residuals = vqvae.quantize.f_to_idxBl_or_fhat_with_residuals(f, used_patch_nums=(1, 4, 8, 16))
            
            gt_wo_first_l = vqvae.quantize.idxBl_to_var_input_with_residuals(gt_tokens, gt_residuals,used_patch_nums=(1, 4, 8, 16))
        
        # Forward pass
        predict_BLV = model(u_avg, gt_wo_first_l)
        predict_BLV = predict_BLV[:, model.first_l:, :]  # Remove conditioning tokens
        
        # Compute loss (ignore conditioning tokens in first_l positions)
        loss, accuracy = compute_loss(predict_BLV, gt_tokens)
        
        # Backward pass
        optimizer.zero_grad()                                                                    
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        total_acc += accuracy.item()
        num_batches += 1
        
        # Print every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch [{batch_idx + 1}/{len(dataloader)}] - "
                  f"Loss: {loss.item():.4f}, Acc: {accuracy.item():.4f}")
    
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches * 100.0
    print(f"  Train Summary - Loss: {avg_loss:.4f}, Acc: {avg_acc:.2f}%")
    
    return avg_loss, avg_acc


def validate(model, vqvae, dataloader, device, patch_nums):
    """Validate for one epoch"""
    model.eval()
    vqvae.eval()
    
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            D_target = batch['D'].to(device)
            u_avg = batch['u'].to(device)

            u_avg = u_avg.unsqueeze(1)  # B×1×16×16 for spatial conditioning
            
            f = vqvae.quant_conv(vqvae.encoder(D_target))
            gt_tokens, gt_residuals = vqvae.quantize.f_to_idxBl_or_fhat_with_residuals(f, used_patch_nums=(1, 4, 8, 16))
            
            gt_wo_first_l = vqvae.quantize.idxBl_to_var_input_with_residuals(gt_tokens, gt_residuals,used_patch_nums=(1, 4, 8, 16))

            predict_BLV = model(u_avg, gt_wo_first_l)
            predict_BLV = predict_BLV[:, model.first_l:, :]  # Remove conditioning tokens
            loss, accuracy = compute_loss(predict_BLV, gt_tokens)
            
            total_loss += loss.item()
            total_acc += accuracy.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches * 100.0
    print(f"  Val Summary - Loss: {avg_loss:.4f}, Acc: {avg_acc:.2f}%")

    return avg_loss, avg_acc

def main():
    # Configuration - Updated to match train.py data paths
    config = {
        'vqvae_path': '/home/ys460/Desktop/Inverse_Problem/VQVAE_generation/vae_ch160v4096z32.pth',
        'train_data_path': '/home/ys460/Desktop/Inverse_Problem/VAR-condition-Fullinp/DiffusionCoefficient_Train5000/grf_data_5000.npz',
        'val_data_path': '/home/ys460/Desktop/Inverse_Problem/VQVAE_generation/DiffusionCoefficient_Test100/grf_data_100.npz',
        'output_dir': './checkpoints_11_20_5000',
        'batch_size': 32,  
        'num_epochs': 250,
        'lr': 5e-4,  
        'weight_decay': 0.05,
        'warmup_epochs': 10, 
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4,
        'save_interval': 5,
        'd_model': 256,
        'nhead': 8,
        'num_layers': 8,
        'dim_feedforward': 512,
    }
    
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Load datasets - Same as train.py
    print("Loading datasets...")
    train_data = np.load(config['train_data_path'])
    val_data = np.load(config['val_data_path'])
    
    D_train = torch.tensor(train_data['D'], dtype=torch.float32)
    u_train = torch.tensor(train_data['u'], dtype=torch.float32)
    
    D_val = torch.tensor(val_data['D'], dtype=torch.float32)
    u_val = torch.tensor(val_data['u'], dtype=torch.float32)
    
    # Average u to 16x16 - Same as train.py
    u_train_avg = image_to_average(u_train, target_size=(16, 16))
    print(f"Averaged u_train shape: {u_train_avg.shape}")
    u_val_avg = image_to_average(u_val, target_size=(16, 16))
    print(f"Averaged u_val shape: {u_val_avg.shape}")
    
    # Load pretrained VQ-VAE
    print("Loading VQ-VAE...")
    vqvae = VQVAE(
        vocab_size=4096,
        z_channels=32,
        ch=160,
        test_mode=True,
    )
    
    checkpoint = torch.load(config['vqvae_path'], map_location='cpu')
    vqvae.load_state_dict(checkpoint, strict=True)
    vqvae = vqvae.to(device)
    vqvae.eval()
    
    print(f"VQ-VAE loaded. Patch numbers: {vqvae.quantize.v_patch_nums}")
    patch_nums = vqvae.quantize.v_patch_nums
    print(f"Using patch numbers for VQVAE: {patch_nums}")
    
    # Create model
    print("Creating autoregressive model...")
    model = PDEVAR(
        vqvae=vqvae,
        patch_nums=(1, 4, 8, 16),
        spatial_cond_size=16,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {num_params:,} trainable parameters")
    
    # Create datasets with PDE data
    print("Creating PDE datasets...")
    train_dataset = PDEDataset(D_train, u_train_avg)
    val_dataset = PDEDataset(D_val, u_val_avg)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.95),
    )
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < config['warmup_epochs']:
            return (epoch + 1) / config['warmup_epochs']
        return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    
    # Lists to store metrics
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"{'='*50}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, vqvae, train_loader, optimizer, device, epoch, patch_nums
        )
        
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        
        # Validate
        val_loss, val_acc = validate(model, vqvae, val_loader, device, patch_nums)
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step()
        
        # Plot curves every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            plot_path = os.path.join(config['output_dir'], 'training_curves.png')
            plot_curves(train_losses, train_accs, val_losses, val_accs, plot_path)
        
        # Save checkpoint
        if epoch % config['save_interval'] == 0:
            checkpoint_path = os.path.join(config['output_dir'], f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_losses': train_losses,
                'train_accs': train_accs,
                'val_losses': val_losses,
                'val_accs': val_accs,
                'config': config,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(config['output_dir'], 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'config': config,
            }, best_path)
            print(f"Best model saved with val_loss: {val_loss:.4f}")
    
    # Final plot
    plot_path = os.path.join(config['output_dir'], 'training_curves_final.png')
    plot_curves(train_losses, train_accs, val_losses, val_accs, plot_path)
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()