import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import math
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
from datetime import datetime

# Import your VQVAE modules (assuming they're in the same directory)
from models.vqvae import VQVAE
from models.basic_vae import Encoder, Decoder
from models.quant import VectorQuantizer2

# ==================== Simple Conv for Stage 1 Feature Extraction ====================
class SimpleConvStage1(nn.Module):
    """Simple Convolution for learning feature extraction from single channel to multi-channel"""
    
    def __init__(self, in_channels=1, out_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        return self.conv(x)

# ==================== Stage2: Transformer-based Inverse Network ====================
class PositionalEncoding2D(nn.Module):
    """Learnable 2D positional encoding for transformer"""
    def __init__(self, channels, height=16, width=16):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        
        # Learnable positional embedding parameters
        self.pos_embedding = nn.Parameter(torch.randn(1, channels, height, width))
        
    def forward(self, tensor):
        B, C, H, W = tensor.shape
        
        # Return the learnable positional embedding, expanded to batch size
        # If input size doesn't match, interpolate
        if H != self.height or W != self.width:
            pos_emb = F.interpolate(self.pos_embedding, size=(H, W), mode='bilinear', align_corners=False)
        else:
            pos_emb = self.pos_embedding
        
        return pos_emb.expand(B, -1, -1, -1)


class TransformerInverseNetwork(nn.Module):
    """Transformer-based network for inverse mapping"""
    def __init__(self, channels=32, embed_dim=256, num_heads=8, num_layers=8, mlp_ratio=4.0, height=16, width=16):
        super().__init__()
        self.channels = channels
        self.embed_dim = embed_dim
        
        # Input projection: channels -> embed_dim
        self.input_proj = nn.Conv2d(channels, embed_dim, 1)
        
        # Positional encoding for embed_dim
        self.pos_encoding = PositionalEncoding2D(embed_dim, height=height, width=width)
        
        self.norm_input = nn.LayerNorm(embed_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])
        
        # Output projection: embed_dim -> channels
        self.output_proj = nn.Conv2d(embed_dim, channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Project to embedding dimension
        x = self.input_proj(x) # (B, embed_dim, H, W)
        
        # Add positional encoding
        pos_enc = self.pos_encoding(x)
        x = x + pos_enc
        
        # Reshape for transformer: (B, embed_dim, H, W) -> (B, H*W, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm_input(x)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Reshape back: (B, H*W, embed_dim) -> (B, embed_dim, H, W)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, H, W)
        
        # Output projection
        x = self.output_proj(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and MLP"""
    def __init__(self, dim, num_heads=8, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


# ==================== Complete Two-Stage Network ====================
class TwoStageInverseNetwork(nn.Module):
    """Complete two-stage network: Simple Conv + Transformer"""
    def __init__(self, in_channels=1, latent_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.stage1_conv = SimpleConvStage1(in_channels, latent_channels)
        self.stage2_transformer = TransformerInverseNetwork(
            channels=latent_channels,
            embed_dim=256,
            num_heads=8,
            num_layers=8,
            mlp_ratio=4.0
        )
        
    def forward(self, x):
        # Stage 1: Learn features using Simple Conv
        features = self.stage1_conv(x)

        # Stage 2: Inverse mapping
        latent_pred = self.stage2_transformer(features)
        
        return latent_pred


# ==================== Dataset and Data Processing ====================
class ImagePredictionDataset(Dataset):
    """Dataset for image prediction task"""
    def __init__(self, D_images, u_averages):
        """
        Args:
            D_images: Full resolution images (N, 256, 256)
            u_averages: Averaged images (N, 16, 16)
        """
        self.D_images = D_images
        self.u_averages = u_averages
        
    def __len__(self):
        return len(self.D_images)
    
    def __getitem__(self, idx):
        return self.D_images[idx], self.u_averages[idx]


def process_u_average(u_average):
    """
    Process u_average from (B, 16, 16) to (B, 1, 16, 16)
    """
    if len(u_average.shape) == 3:  # (B, 16, 16)
        u_average = u_average.unsqueeze(1)  # (B, 1, 16, 16)
    return u_average


def image_to_average(image, target_size=(16, 16)):
    '''
    input: image (B, H, W) or (H, W)
    output: averaged image (B, target_H, target_W) or (target_H, target_W)
    '''
    if len(image.shape) == 2:
        image = image.unsqueeze(0)  # (1, H, W)
    
    B, H, W = image.shape
    target_H, target_W = target_size
    assert H % target_H == 0 and W % target_W == 0, "Image dimensions must be divisible by target size"
    block_H, block_W = H // target_H, W // target_W
    image = image.view(B, target_H, block_H, target_W, block_W)
    averaged = image.mean(dim=(2, 4))  # (B, target_H, target_W)
    return averaged


# ==================== Training and Inference Pipeline ====================
class VQVAEImagePredictionPipeline:
    """Complete pipeline for image prediction using VQVAE"""
    
    def __init__(self, vqvae_path, device='cuda', checkpoint_dir='checkpoints_trans_nasty35'):
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Training history for plotting
        self.train_losses = []
        self.val_losses = []
        
        # Load pretrained VQVAE
        print("Loading pretrained VQVAE...")
        self.vqvae = self.load_vqvae(vqvae_path)
        self.vqvae.eval()
        
        # Initialize two-stage network
        self.model = TwoStageInverseNetwork(in_channels=1, latent_channels=32).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=5e-4, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
    def load_vqvae(self, path):
        """Load pretrained VQVAE model"""
        # Initialize VQVAE with your configuration
        vqvae = VQVAE(
            vocab_size=4096,
            z_channels=32,
            ch=160,
            v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
            test_mode=True
        ).to(self.device)
        
        # Load weights
        checkpoint = torch.load(path, map_location=self.device)
        if 'model' in checkpoint:
            vqvae.load_state_dict(checkpoint['model'])
        else:
            vqvae.load_state_dict(checkpoint)
        
        return vqvae
    
    def encode_images(self, images):
        """Encode images to VQVAE latent space"""
        with torch.no_grad():
            # Ensure images have batch and channel dimensions
            if len(images.shape) == 3:  # (B, H, W)
                images = images.unsqueeze(1)  # (B, 1, H, W)
            
            # Expand to 3 channels if single channel
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)  # (B, 3, H, W)
            
            # Normalize images from [0, 1] to [-1, 1]
            images = (images * 2) - 1
            
            # Encode through VQVAE encoder
            encoded = self.vqvae.quant_conv(self.vqvae.encoder(images))
            return encoded
    
    def decode_latents(self, latents):
        """Decode latents back to images"""
        with torch.no_grad():
            # Decode through VQVAE decoder
            decoded = self.vqvae.decoder(self.vqvae.post_quant_conv(latents))
            # Clamp to valid range [-1, 1]
            decoded = decoded.clamp(-1, 1)
            # Convert from [-1, 1] to [0, 1] range
            decoded = (decoded + 1) / 2
            # Take only first channel
            decoded = decoded[:, 0:1, :, :]  # (B, 1, H, W)
            return decoded

    def plot_training_curves(self, save_path='training_curves.png'):
        """Plot training and validation curves with log scale"""
        if len(self.train_losses) == 0:
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(self.train_losses) + 1)
        ax.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2, marker='o', markersize=3)
        
        if len(self.val_losses) > 0:
            ax.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=3)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (log scale)', fontsize=12)
        ax.set_title('Training Progress', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Set log scale for y-axis
        ax.set_yscale('log')
        
        # Add min loss annotations
        if len(self.train_losses) > 0:
            min_train_epoch = np.argmin(self.train_losses) + 1
            min_train_loss = np.min(self.train_losses)
            ax.annotate(f'Min: {min_train_loss:.4f}', 
                       xy=(min_train_epoch, min_train_loss),
                       xytext=(min_train_epoch+5, min_train_loss*1.5),
                       arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7),
                       fontsize=9, color='blue')
        
        if len(self.val_losses) > 0:
            min_val_epoch = np.argmin(self.val_losses) + 1
            min_val_loss = np.min(self.val_losses)
            ax.annotate(f'Min: {min_val_loss:.4f}', 
                       xy=(min_val_epoch, min_val_loss),
                       xytext=(min_val_epoch+5, min_val_loss*0.7),
                       arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                       fontsize=9, color='red')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to {save_path}")

    def train_step(self, D_images, u_averages):
        """Single training step"""
        self.model.train()
        
        # Move to device
        D_images = D_images.to(self.device)
        u_averages = u_averages.to(self.device)
        
        # Process u_averages
        u_input = process_u_average(u_averages)
        
        # Get target latents from D_images
        with torch.no_grad():
            D_latents = self.encode_images(D_images)
        
        # Forward pass through model
        D_latents_pred = self.model(u_input)
        
        # Compute loss
        loss = F.mse_loss(D_latents_pred, D_latents)
        
        # Add regularization loss
        reg_loss = 0.01 * torch.mean(torch.abs(D_latents_pred))
        total_loss = loss + reg_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'reg_loss': reg_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def validate(self, val_loader):
        """Validation step"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for D_images, u_averages in val_loader:
                D_images = D_images.to(self.device)
                u_averages = u_averages.to(self.device)
                
                u_input = process_u_average(u_averages)
                D_latents = self.encode_images(D_images)
                D_latents_pred = self.model(u_input)
                
                loss = F.mse_loss(D_latents_pred, D_latents)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader=None, num_epochs=100):
        """Full training loop with periodic checkpointing and plotting"""
        best_val_loss = float('inf')
        
        # Clear training history
        self.train_losses = []
        self.val_losses = []
        
        for epoch in range(num_epochs):
            # Training
            epoch_losses = []
            for batch_idx, (D_images, u_averages) in enumerate(train_loader):
                losses = self.train_step(D_images, u_averages)
                epoch_losses.append(losses)
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                          f"Loss: {losses['loss']:.4f}")
            
            # Calculate average training loss
            avg_train_loss = np.mean([l['loss'] for l in epoch_losses])
            self.train_losses.append(avg_train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint('best_model.pth', epoch+1, avg_train_loss, val_loss)
                    print(f"  -> New best model saved (Val Loss: {val_loss:.4f})")
            else:
                print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_name = f'checkpoint_epoch_{epoch+1}.pth'
                checkpoint_path = self.checkpoint_dir / checkpoint_name
                self.save_checkpoint(
                    checkpoint_path, 
                    epoch+1, 
                    avg_train_loss, 
                    val_loss if val_loader is not None else None
                )
                print(f"  -> Checkpoint saved: {checkpoint_name}")
                
                # Plot training curves
                self.plot_training_curves(self.checkpoint_dir / f'training_curves_epoch_{epoch+1}.png')
                
                # Also save training history
                self.save_training_history(self.checkpoint_dir / 'training_history.npz')
            
            # Update learning rate
            self.scheduler.step()
            
        # Final save
        print("\nTraining completed!")
        self.save_checkpoint(
            self.checkpoint_dir / 'final_model.pth',
            num_epochs,
            self.train_losses[-1],
            self.val_losses[-1] if val_loader is not None else None
        )
        self.plot_training_curves(self.checkpoint_dir / 'training_curves_final.png')
        self.save_training_history(self.checkpoint_dir / 'training_history.npz')

    def save_checkpoint(self, path, epoch=None, train_loss=None, val_loss=None):
        """Save model checkpoint with metadata"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_losses': torch.tensor(self.train_losses) if self.train_losses else torch.tensor([]),
            'val_losses': torch.tensor(self.val_losses) if self.val_losses else torch.tensor([]),
            'timestamp': datetime.now().isoformat()
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training history if available
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        
        print(f"Checkpoint loaded from {path}")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'train_loss' in checkpoint:
            print(f"  Train Loss: {checkpoint['train_loss']:.4f}")
        if 'val_loss' in checkpoint and checkpoint['val_loss'] is not None:
            print(f"  Val Loss: {checkpoint['val_loss']:.4f}")

    def save_training_history(self, path):
        """Save training history to file"""
        np.savez(path,
                 train_losses=np.array(self.train_losses),
                 val_losses=np.array(self.val_losses))
        print(f"Training history saved to {path}")
    
    def load_training_history(self, path):
        """Load training history from file"""
        data = np.load(path)
        self.train_losses = data['train_losses'].tolist()
        self.val_losses = data['val_losses'].tolist()
        print(f"Training history loaded from {path}")
    
    def predict(self, u_averages):
        """Predict full images from averages"""
        self.model.eval()
        
        with torch.no_grad():
            # Process input
            u_averages = u_averages.to(self.device)
            u_input = process_u_average(u_averages)
            
            # Predict latents
            D_latents_pred = self.model(u_input)
            
            # Decode to images
            D_images_pred = self.decode_latents(D_latents_pred)
            
        return D_images_pred
    
    def evaluate(self, test_loader):
        """Evaluate model performance"""
        self.model.eval()
        
        total_mse = 0
        total_psnr = 0
        num_samples = 0
        
        with torch.no_grad():
            for D_images, u_averages in test_loader:
                # Predict
                D_images_pred = self.predict(u_averages)
                D_images = D_images.to(self.device)
                
                # Ensure D_images has channel dimension for comparison
                if len(D_images.shape) == 3:  # (B, H, W)
                    D_images = D_images.unsqueeze(1)  # (B, 1, H, W)
                
                # Both images should be in [0, 1] range
                # Calculate metrics
                mse = F.mse_loss(D_images_pred, D_images).item()
                
                total_mse += mse * D_images.shape[0]
                num_samples += D_images.shape[0]
        
        avg_mse = total_mse / num_samples
        
        print(f"Evaluation Results:")
        print(f"  Average MSE: {avg_mse:.6f}")

        return {'mse': avg_mse}


def main():
    # Set paths and parameters
    vqvae_path = "/home/ys460/Desktop/Inverse_Problem/VQVAE_generation/vae_ch160v4096z32.pth"

    train_path = "/home/ys460/Desktop/Inverse_Problem/VQVAE_generation/DiffusionCoefficient_nasty35_Train1000/grf_data_1000.npz"
    val_path = "/home/ys460/Desktop/Inverse_Problem/VQVAE_generation/DiffusionCoefficient_nasty35_Test100/grf_data_100.npz"
    
    # Checkpoint directory
    checkpoint_dir = 'checkpoints_trans_nasty35'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 16
    num_epochs = 100

    # Initialize pipeline
    pipeline = VQVAEImagePredictionPipeline(vqvae_path, device, checkpoint_dir=checkpoint_dir)

    # Load datasets
    train_data = np.load(train_path)
    val_data = np.load(val_path)

    D_train = torch.tensor(train_data['D'], dtype=torch.float32)
    u_train = torch.tensor(train_data['u'], dtype=torch.float32)
    print(f"Train D shape: {D_train.shape}, u shape: {u_train.shape}")
    D_val = torch.tensor(val_data['D'], dtype=torch.float32)
    u_val = torch.tensor(val_data['u'], dtype=torch.float32)
    print(f"Val D shape: {D_val.shape}, u shape: {u_val.shape}")

    # Average u_train to 16x16
    u_train_avg = image_to_average(u_train, target_size=(16, 16))
    print(f"Averaged u_train shape: {u_train_avg.shape}")
    u_val_avg = image_to_average(u_val, target_size=(16, 16))
    print(f"Averaged u_val shape: {u_val_avg.shape}")

    # Process datasets
    train_dataset = ImagePredictionDataset(
        D_train,
        u_train_avg
    )

    val_dataset = ImagePredictionDataset(
        D_val,
        u_val_avg
    )

    test_dataset = ImagePredictionDataset(
        D_val,
        u_val_avg
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Train model
    print("Starting training...")
    pipeline.train(train_loader, val_loader, num_epochs)

    # Evaluate model
    print("\nEvaluating model...")
    pipeline.evaluate(test_loader)

    # Example prediction
    print("\nExample prediction...")
    sample_u = u_val_avg[:4]  # Take 4 samples
    predicted_images = pipeline.predict(sample_u)
    print(f"Predicted images shape: {predicted_images.shape}")


if __name__ == "__main__":
    main()