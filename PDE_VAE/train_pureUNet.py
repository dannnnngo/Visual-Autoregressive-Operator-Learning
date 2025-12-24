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

# ==================== Enhanced U-Net for Direct Latent Prediction ====================
class EnhancedUNet(nn.Module):
    """Enhanced U-Net for direct mapping from u (1, 16, 16) to latent space (32, 16, 16)"""
    
    def __init__(self, in_channels=1, out_channels=32, features=[64, 128, 256]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True)
        )
        
        # Encoder path (without pooling since input is already 16x16)
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(features)):
            in_feat = features[0] if i == 0 else features[i-1]
            out_feat = features[i]
            self.encoder_blocks.append(self.conv_block(in_feat, out_feat))
        
        # Bottleneck
        self.bottleneck = self.conv_block(features[-1], features[-1] * 2)
        
        # Decoder path
        self.decoder_blocks = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        
        for i in range(len(features)-1, -1, -1):
            in_feat = features[-1] * 2 if i == len(features)-1 else features[i+1]
            out_feat = features[i]
            
            # Upsampling is replaced with feature refinement since we maintain spatial resolution
            self.upconvs.append(
                nn.Sequential(
                    nn.Conv2d(in_feat, out_feat, kernel_size=1),
                    nn.BatchNorm2d(out_feat),
                    nn.ReLU(inplace=True)
                )
            )
            # Decoder block takes skip connection
            self.decoder_blocks.append(self.conv_block(out_feat * 2, out_feat))
        
        # Final layers with residual connections
        self.final_layers = nn.Sequential(
            nn.Conv2d(features[0], features[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0], out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )
        
    def conv_block(self, in_channels, out_channels):
        """Convolutional block with BatchNorm and ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Initial feature extraction
        x = self.init_conv(x)
        
        # Encoder path - store skip connections
        encoder_features = []
        for encoder in self.encoder_blocks:
            x = encoder(x)
            encoder_features.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with skip connections
        for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoder_blocks)):
            x = upconv(x)
            # Get corresponding encoder feature (reversed order)
            skip_connection = encoder_features[-(i+1)]
            x = torch.cat([x, skip_connection], dim=1)
            x = decoder(x)
        
        # Final output
        output = self.final_layers(x)
        
        return output


class DeepUNet(nn.Module):
    """Deeper U-Net with attention mechanisms for better feature extraction"""
    
    def __init__(self, in_channels=1, out_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Feature expansion path
        self.expand1 = self.conv_block(in_channels, 64)
        self.expand2 = self.conv_block(64, 128)
        self.expand3 = self.conv_block(128, 256)
        
        # Attention gates
        self.attention1 = SelfAttention(64)
        self.attention2 = SelfAttention(128)
        self.attention3 = SelfAttention(256)
        
        # Central processing
        self.center = nn.Sequential(
            self.conv_block(256, 512),
            self.conv_block(512, 512),
            self.conv_block(512, 256)
        )
        
        # Feature compression path
        self.compress3 = self.conv_block(512, 256)  # 256 from skip + 256 from center
        self.compress2 = self.conv_block(384, 128)  # 128 from skip + 256 from previous
        self.compress1 = self.conv_block(192, 64)   # 64 from skip + 128 from previous
        
        # Final projection
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Expansion path with attention
        e1 = self.expand1(x)
        e1_att = self.attention1(e1)
        
        e2 = self.expand2(e1_att)
        e2_att = self.attention2(e2)
        
        e3 = self.expand3(e2_att)
        e3_att = self.attention3(e3)
        
        # Center processing
        center = self.center(e3_att)
        
        # Compression path with skip connections
        d3 = torch.cat([center, e3_att], dim=1)
        d3 = self.compress3(d3)
        
        d2 = torch.cat([d3, e2_att], dim=1)
        d2 = self.compress2(d2)
        
        d1 = torch.cat([d2, e1_att], dim=1)
        d1 = self.compress1(d1)
        
        # Final output
        output = self.final_conv(d1)
        
        return output


class SelfAttention(nn.Module):
    """Self-attention module for feature refinement"""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.query_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Generate query, key, value
        query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        key = self.key_conv(x).view(B, -1, H * W)
        value = self.value_conv(x).view(B, C, H * W)
        
        # Compute attention
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to value
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        
        # Residual connection with learnable weight
        out = self.gamma * out + x
        
        return out


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
class PureUNetPipeline:
    """Pipeline for direct U-Net based latent prediction"""

    def __init__(self, vqvae_path, device='cuda', checkpoint_dir='checkpoints_deepunet_nasty35', use_deep_unet=False):
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
        
        # Initialize U-Net model
        if use_deep_unet:
            print("Using Deep U-Net with attention mechanisms")
            self.model = DeepUNet(in_channels=1, out_channels=32).to(device)
        else:
            print("Using Enhanced U-Net")
            self.model = EnhancedUNet(in_channels=1, out_channels=32).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Initialize optimizer with gradient accumulation support
        self.optimizer = optim.AdamW(self.model.parameters(), lr=5e-4, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
    def load_vqvae(self, path):
        """Load pretrained VQVAE model"""
        vqvae = VQVAE(
            vocab_size=4096,
            z_channels=32,
            ch=160,
            v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
            test_mode=True
        ).to(self.device)
        
        checkpoint = torch.load(path, map_location=self.device)
        if 'model' in checkpoint:
            vqvae.load_state_dict(checkpoint['model'])
        else:
            vqvae.load_state_dict(checkpoint)
        
        return vqvae
    
    def encode_images(self, images):
        """Encode images to VQVAE latent space"""
        with torch.no_grad():
            if len(images.shape) == 3:  # (B, H, W)
                images = images.unsqueeze(1)  # (B, 1, H, W)
            
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)  # (B, 3, H, W)
            
            images = (images * 2) - 1
            encoded = self.vqvae.quant_conv(self.vqvae.encoder(images))
            return encoded
    
    def decode_latents(self, latents):
        """Decode latents back to images"""
        with torch.no_grad():
            decoded = self.vqvae.decoder(self.vqvae.post_quant_conv(latents))
            decoded = decoded.clamp(-1, 1)
            decoded = (decoded + 1) / 2
            decoded = decoded[:, 0:1, :, :]  # (B, 1, H, W)
            return decoded

    def train_step(self, D_images, u_averages):
        """Single training step with mixed precision training"""
        self.model.train()
        
        D_images = D_images.to(self.device)
        u_averages = u_averages.to(self.device)
        
        # Process u_averages
        u_input = process_u_average(u_averages)
        
        # Get target latents from D_images
        with torch.no_grad():
            D_latents = self.encode_images(D_images)
        
        # Forward pass through U-Net
        D_latents_pred = self.model(u_input)
        
        # Compute losses
        # Main reconstruction loss
        mse_loss = F.mse_loss(D_latents_pred, D_latents)
        
        # Additional L1 loss for better convergence
        l1_loss = F.l1_loss(D_latents_pred, D_latents)
        
        # Feature matching loss (optional)
        # This helps preserve structure
        cosine_loss = 1 - F.cosine_similarity(
            D_latents_pred.view(D_latents_pred.size(0), -1),
            D_latents.view(D_latents.size(0), -1),
            dim=1
        ).mean()
        
        # Total loss
        total_loss = mse_loss + 0.1 * l1_loss + 0.05 * cosine_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'loss': mse_loss.item(),
            'l1_loss': l1_loss.item(),
            'cosine_loss': cosine_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def validate(self, val_loader):
        """Validation step"""
        self.model.eval()
        total_loss = 0
        total_l1_loss = 0
        
        with torch.no_grad():
            for D_images, u_averages in val_loader:
                D_images = D_images.to(self.device)
                u_averages = u_averages.to(self.device)
                
                u_input = process_u_average(u_averages)
                D_latents = self.encode_images(D_images)
                D_latents_pred = self.model(u_input)
                
                loss = F.mse_loss(D_latents_pred, D_latents)
                l1_loss = F.l1_loss(D_latents_pred, D_latents)
                
                total_loss += loss.item()
                total_l1_loss += l1_loss.item()
        
        avg_loss = total_loss / len(val_loader)
        avg_l1_loss = total_l1_loss / len(val_loader)
        
        return avg_loss, avg_l1_loss
    
    def train(self, train_loader, val_loader=None, num_epochs=100):
        """Full training loop"""
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 20
        
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
                          f"MSE: {losses['loss']:.4f}, L1: {losses['l1_loss']:.4f}")
            
            # Calculate average training loss
            avg_train_loss = np.mean([l['loss'] for l in epoch_losses])
            self.train_losses.append(avg_train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss, val_l1_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val L1: {val_l1_loss:.4f}")
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(self.checkpoint_dir / 'best_model.pth', epoch+1, avg_train_loss, val_loss)
                    print(f"  -> New best model saved (Val Loss: {val_loss:.4f})")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                # Early stopping
                if patience_counter >= max_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
            else:
                print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_name = f'checkpoint_epoch_{epoch+1}.pth'
                self.save_checkpoint(
                    self.checkpoint_dir / checkpoint_name,
                    epoch+1,
                    avg_train_loss,
                    val_loss if val_loader is not None else None
                )
                self.plot_training_curves(self.checkpoint_dir / f'training_curves_epoch_{epoch+1}.png')
                self.save_training_history(self.checkpoint_dir / 'training_history.npz')
        
        # Final save
        print("\nTraining completed!")
        self.save_checkpoint(
            self.checkpoint_dir / 'final_model.pth',
            epoch+1,
            self.train_losses[-1],
            self.val_losses[-1] if val_loader is not None else None
        )
        self.plot_training_curves(self.checkpoint_dir / 'training_curves_final.png')
    
    def plot_training_curves(self, save_path='training_curves.png'):
        """Plot training and validation curves"""
        if len(self.train_losses) == 0:
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(self.train_losses) + 1)
        ax.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2, marker='o', markersize=3)
        
        if len(self.val_losses) > 0:
            ax.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=3)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (log scale)', fontsize=12)
        ax.set_title('Pure U-Net Training Progress', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to {save_path}")

    def save_checkpoint(self, path, epoch=None, train_loss=None, val_loss=None):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
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
        
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        
        print(f"Checkpoint loaded from {path}")

    def save_training_history(self, path):
        """Save training history"""
        np.savez(path,
                 train_losses=np.array(self.train_losses),
                 val_losses=np.array(self.val_losses))
    
    def predict(self, u_averages):
        """Predict full images from averages"""
        self.model.eval()
        
        with torch.no_grad():
            u_averages = u_averages.to(self.device)
            u_input = process_u_average(u_averages)
            
            # Predict latents using U-Net
            D_latents_pred = self.model(u_input)
            
            # Decode to images
            D_images_pred = self.decode_latents(D_latents_pred)
            
        return D_images_pred
    
    def evaluate(self, test_loader):
        """Evaluate model performance"""
        self.model.eval()
        
        total_mse = 0
        num_samples = 0
        
        with torch.no_grad():
            for D_images, u_averages in test_loader:
                D_images_pred = self.predict(u_averages)
                D_images = D_images.to(self.device)
                
                if len(D_images.shape) == 3:
                    D_images = D_images.unsqueeze(1)
                
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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 16
    num_epochs = 100

    # Initialize pipeline with Deep U-Net
    pipeline = PureUNetPipeline(vqvae_path, device, use_deep_unet=True)

    # Load datasets
    train_data = np.load(train_path)
    val_data = np.load(val_path)

    D_train = torch.tensor(train_data['D'], dtype=torch.float32)
    u_train = torch.tensor(train_data['u'], dtype=torch.float32)
    print(f"Train D shape: {D_train.shape}, u shape: {u_train.shape}")
    
    D_val = torch.tensor(val_data['D'], dtype=torch.float32)
    u_val = torch.tensor(val_data['u'], dtype=torch.float32)
    print(f"Val D shape: {D_val.shape}, u shape: {u_val.shape}")

    # Average u to 16x16
    u_train_avg = image_to_average(u_train, target_size=(16, 16))
    u_val_avg = image_to_average(u_val, target_size=(16, 16))
    print(f"Averaged u_train shape: {u_train_avg.shape}")
    print(f"Averaged u_val shape: {u_val_avg.shape}")

    # Create datasets
    train_dataset = ImagePredictionDataset(D_train, u_train_avg)
    val_dataset = ImagePredictionDataset(D_val, u_val_avg)
    test_dataset = ImagePredictionDataset(D_val, u_val_avg)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Train model
    print("\nStarting training with Pure U-Net...")
    pipeline.train(train_loader, val_loader, num_epochs)

    # Evaluate model
    print("\nEvaluating model...")
    pipeline.evaluate(test_loader)

    # Example prediction
    print("\nExample prediction...")
    sample_u = u_val_avg[:4]
    predicted_images = pipeline.predict(sample_u)
    print(f"Predicted images shape: {predicted_images.shape}")


if __name__ == "__main__":
    main()