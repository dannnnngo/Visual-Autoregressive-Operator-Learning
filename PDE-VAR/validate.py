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
import torch.distributed as dist

# Import VQVAE modules
from models.vqvae import VQVAE
from models.basic_vae import Encoder, Decoder
from models.quant import VectorQuantizer2
from models.pdevar import PDEVAR

import train_var


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
        
        # Normalize D to [-1, 1] if not already
        if D_img_3ch.min() >= 0 and D_img_3ch.max() <= 1:
            D_img_3ch = 2.0 * D_img_3ch - 1.0
        
        return {
            'D': D_img_3ch,  # High-resolution D field (target) [3, 256, 256], range [-1, 1]
            'u': u_avg,  # Low-resolution u field (conditioning) [16, 16], range [0, 1]
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

def main():
    config = {
        'vqvae_path': '/home/ys460/Desktop/Inverse_Problem/VQVAE_generation/vae_ch160v4096z32.pth',
        'val_data_path': '/home/ys460/Desktop/Inverse_Problem/VQVAE_generation/DiffusionCoefficient_Test100/grf_data_100.npz',
        'checkpoint_path': '/home/ys460/Desktop/Inverse_Problem/VAR-condition-Fullinp-OwnCode_attempt2/checkpoints_11_20_5000/best_model.pt', 
        'output_dir': './validation_results_1120_5000_4patches',
        'batch_size': 2, 
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4,
        'd_model': 256,
        'nhead': 8,
        'num_layers': 8, 
        'dim_feedforward': 512, 
    }
    
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    val_data = np.load(config['val_data_path'])
    
    D_val = torch.tensor(val_data['D'], dtype=torch.float32)
    u_val = torch.tensor(val_data['u'], dtype=torch.float32)
    
    # Average u to 16x16
    u_val_avg = image_to_average(u_val, target_size=(16, 16))
    
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

    patch_nums = vqvae.quantize.v_patch_nums

    # Load config from checkpoint if available
    checkpoint_path = config['checkpoint_path']
    print(f"Loading model checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Override config with saved config if available
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        config['d_model'] = saved_config.get('d_model', config['d_model'])
        config['nhead'] = saved_config.get('nhead', config['nhead'])
        config['num_layers'] = saved_config.get('num_layers', config['num_layers'])
        config['dim_feedforward'] = saved_config.get('dim_feedforward', config['dim_feedforward'])
        print(f"Using model config from checkpoint: d_model={config['d_model']}, "
              f"nhead={config['nhead']}, num_layers={config['num_layers']}, "
              f"dim_feedforward={config['dim_feedforward']}")

    model = PDEVAR(
        vqvae=vqvae,
        patch_nums=(1, 4, 8, 16),
        spatial_cond_size=16,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
    ).to(device)

    # Handle different checkpoint formats
    print(f"Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dict'}")
    
    if isinstance(checkpoint, dict):
        # Check for different possible key names
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            # Assume the checkpoint dict itself is the state_dict
            sample_key = list(checkpoint.keys())[0] if checkpoint else None
            if sample_key and isinstance(checkpoint[sample_key], torch.Tensor):
                state_dict = checkpoint
            else:
                raise KeyError(f"Cannot find model state dict. Available keys: {list(checkpoint.keys())}")
    else:
        state_dict = checkpoint
    
    # Try loading with strict=True first
    try:
        model.load_state_dict(state_dict, strict=True)
        print("Model loaded successfully")
    except RuntimeError as e:
        print(f"Warning: {e}")
        print("Attempting to load with strict=False...")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
    
    model.eval()

    val_dataset = PDEDataset(D_val, u_val_avg)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True)

    # ========== VQ-VAE QUALITY CHECK 
    print("Testing VQ-VAE Reconstruction Quality")
    with torch.no_grad():
        test_batch = next(iter(val_loader)) 
        D_image = test_batch['D'].to(device)  # (B, 3, 256, 256)
        B = D_image.shape[0]
        # Get discrete tokens using img_to_idxBl
        gt_ms_idx_Bl = vqvae.img_to_idxBl(D_image)
        
        # Method 1: Reconstruct from tokens using tokens_to_image
        ms_h_BChw = []
        for scale_idx, pn in zip(gt_ms_idx_Bl, vqvae.quantize.v_patch_nums):
            h_BCnn = vqvae.quantize.embedding(scale_idx)  # (B, L, Cvae)  # Updated to use vqvae
            h_BCnn = h_BCnn.transpose(1, 2).view(B, vqvae.quantize.Cvae, pn, pn)
            ms_h_BChw.append(h_BCnn)
        
        # Combine multi-scale features
        f_hat = vqvae.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=True, last_one=True)
        
        # Decode to image
        decoded_quant = vqvae.post_quant_conv(f_hat)
        reconstructed_from_tokens = vqvae.decoder(decoded_quant)

    # Plot the reconstructed images vs ground truth
    num_samples = min(4, B)
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    for i in range(num_samples):
        gt_img = D_image[i].cpu().numpy().transpose(1, 2, 0)  # (256, 256, 3)
        recon_img = reconstructed_from_tokens[i].cpu().numpy().transpose(1, 2, 0)  # (256, 256, 3)
        
        axes[i, 0].imshow((gt_img + 1) / 2)  # Rescale to [0, 1]
        axes[i, 0].set_title('Ground Truth D')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow((recon_img + 1) / 2)  # Rescale to [0, 1]
        axes[i, 1].set_title('Reconstructed from Tokens')
        axes[i, 1].axis('off')    
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['output_dir'], 'vqvae_reconstruction_check.png'), dpi=150, bbox_inches='tight')
    # ========== END VQ-VAE QUALITY CHECK

    # Validation Loop
    model.eval()
    all_predictions = []
    all_targets = []
    all_conditions = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validating")):
            D_gt = batch['D'].to(device)  # (B, 3, 256, 256)
            u_cond = batch['u'].to(device)  # (B, 16, 16) - already normalized in dataset
            B = u_cond.shape[0]  # Define batch size
            
            # Add channel dimension for CNN encoder in the model
            u_cond_expanded = u_cond.unsqueeze(1)  # (B, 1, 16, 16)
            
            # Generate predictions using autoregressive generation
            generated_tokens, f_hat = model.autoregressive_generate(u_cond_expanded)
            
            # f_hat is already the accumulated multi-scale features in (B, Cvae, 16, 16)
            # We just need to decode it to image
            decoded_quant = vqvae.post_quant_conv(f_hat)
            D_pred = vqvae.decoder(decoded_quant)

            # Store results for batch processing
            all_predictions.append(D_pred.cpu())
            all_targets.append(D_gt.cpu())
            all_conditions.append(u_cond.cpu())  # Store without channel dim
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)  # (N, 3, 256, 256)
    all_targets = torch.cat(all_targets, dim=0)  # (N, 3, 256, 256)
    all_conditions = torch.cat(all_conditions, dim=0)  # (N, 16, 16)
    
    # Compute metrics
    mse = F.mse_loss(all_predictions, all_targets).item()
    mae = F.l1_loss(all_predictions, all_targets).item()
    
    # Compute per-sample metrics
    per_sample_mse = ((all_predictions - all_targets) ** 2).mean(dim=(1, 2, 3))
    per_sample_mae = torch.abs(all_predictions - all_targets).mean(dim=(1, 2, 3))
    
    # Compute L2 relative error per sample
    # L2 relative error = ||pred - target||_2 / ||target||_2
    per_sample_l2_error = torch.sqrt(((all_predictions - all_targets) ** 2).sum(dim=(1, 2, 3))) / torch.sqrt((all_targets ** 2).sum(dim=(1, 2, 3)))
    
    print(f"\nValidation Results:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Mean per-sample MSE: {per_sample_mse.mean():.6f} ± {per_sample_mse.std():.6f}")
    print(f"Mean per-sample MAE: {per_sample_mae.mean():.6f} ± {per_sample_mae.std():.6f}")
    print(f"Mean L2 relative error: {per_sample_l2_error.mean():.6f} ± {per_sample_l2_error.std():.6f}")
    
    # Visualize samples
    num_samples = min(8, len(all_predictions))
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Take first channel for visualization
        condition = all_conditions[i].squeeze().numpy()  # (16, 16)
        target = all_targets[i, 0].numpy()  # (256, 256)
        prediction = all_predictions[i, 0].numpy()  # (256, 256)
        error = np.abs(target - prediction)  # (256, 256)
        
        # Column 1: Condition (u field)
        im0 = axes[i, 0].imshow(condition, cmap='viridis')
        axes[i, 0].set_title(f'Sample {i}\nCondition u')
        axes[i, 0].axis('off')
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046)
        
        # Column 2: Ground Truth (D field)
        im1 = axes[i, 1].imshow(target, cmap='viridis')
        axes[i, 1].set_title(f'Ground Truth D')
        axes[i, 1].axis('off')
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)
        
        # Column 3: Predicted D
        im2 = axes[i, 2].imshow(prediction, cmap='viridis')
        axes[i, 2].set_title(f'Predicted D')
        axes[i, 2].axis('off')
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)
        
        # Column 4: Error Map
        im3 = axes[i, 3].imshow(error, cmap='hot')
        axes[i, 3].set_title(f'Error Map\nMSE: {((target - prediction)**2).mean():.4f}')
        axes[i, 3].axis('off')
        plt.colorbar(im3, ax=axes[i, 3], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['output_dir'], 'validation_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to {config['output_dir']}/validation_comparison.png")
    plt.close()
    
    # Save numerical results
    results = {
        'mse': mse,
        'mae': mae,
        'per_sample_mse': per_sample_mse.numpy(),
        'per_sample_mae': per_sample_mae.numpy(),
        'per_sample_l2_error': per_sample_l2_error.numpy(),
        'predictions': all_predictions.numpy(),
        'targets': all_targets.numpy(),
        'conditions': all_conditions.numpy(),
    }
    np.savez(
        os.path.join(config['output_dir'], 'validation_results.npz'),
        **results
    )
    print(f"Saved numerical results to {config['output_dir']}/validation_results.npz")
    print("\nValidation complete!")


if __name__ == '__main__':
    main()