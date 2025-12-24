import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Optional
import torchvision.transforms.functional as TF
from PIL import Image


def image_to_average(image, target_size=(16, 16)):
    '''
    Downsample image by averaging patches
    input: image (C, H, W) or (B, C, H, W)
    output: averaged image (C, target_H, target_W) or (B, C, target_H, target_W)
    '''
    if len(image.shape) == 3:
        # Single image (C, H, W)
        C, H, W = image.shape
        target_H, target_W = target_size
        assert H % target_H == 0 and W % target_W == 0, "Image dimensions must be divisible by target size"
        
        block_H, block_W = H // target_H, W // target_W
        # Reshape to group patches
        image = image.view(C, target_H, block_H, target_W, block_W)
        # Average over patch dimensions
        averaged = image.mean(dim=(2, 4))  # (C, target_H, target_W)
        return averaged
    
    elif len(image.shape) == 4:
        # Batch of images (B, C, H, W)
        B, C, H, W = image.shape
        target_H, target_W = target_size
        assert H % target_H == 0 and W % target_W == 0, "Image dimensions must be divisible by target size"
        
        block_H, block_W = H // target_H, W // target_W
        # Reshape to group patches
        image = image.view(B, C, target_H, block_H, target_W, block_W)
        # Average over patch dimensions
        averaged = image.mean(dim=(3, 5))  # (B, C, target_H, target_W)
        return averaged
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")


class PDEDataset(Dataset):
    """Dataset for PDE inverse problem: D (target) and u (condition)"""
    
    def __init__(self, data_path: str, target_size: int = 256, normalize: bool = True,
                 condition_patch_size: int = 16):
        """
        Args:
            data_path: Path to .npz file containing D and u arrays
            target_size: Size to resize images to
            normalize: Whether to normalize data to [0, 1]
            condition_patch_size: Size of condition patches (16 for 16x16 patches from 256x256)
        """
        # Load data
        data = np.load(data_path)
        self.D_data = data['D']  # Diffusion coefficient
        self.u_data = data['u']  # Solution field (will be averaged to patches)
        
        # Check dimensions
        print(f"Loaded data from {data_path}")
        print(f"  D shape: {self.D_data.shape}")
        print(f"  u shape: {self.u_data.shape}")
        
        # Handle different input formats
        if len(self.D_data.shape) == 3:  # (N, H, W)
            # Add channel dimension
            self.D_data = self.D_data[:, np.newaxis, :, :]
            self.u_data = self.u_data[:, np.newaxis, :, :]
        elif len(self.D_data.shape) == 4:  # (N, C, H, W)
            pass
        else:
            raise ValueError(f"Unexpected data shape: D={self.D_data.shape}")
        
        self.target_size = target_size
        self.condition_patch_size = condition_patch_size
        self.normalize = normalize
        
        # Normalize data
        if self.normalize:
            # Normalize D and u to [0, 1]
            self.D_min, self.D_max = self.D_data.min(), self.D_data.max()
            self.u_min, self.u_max = self.u_data.min(), self.u_data.max()
            
            self.D_data = (self.D_data - self.D_min) / (self.D_max - self.D_min + 1e-8)
            self.u_data = (self.u_data - self.u_min) / (self.u_max - self.u_min + 1e-8)
            
            print(f"  Normalized D to range [0, 1]")
            print(f"  Normalized u to range [0, 1]")
        
        # Convert to torch tensors
        self.D_data = torch.from_numpy(self.D_data).float()
        self.u_data = torch.from_numpy(self.u_data).float()
        
        # Ensure we have 3 channels for VAE compatibility
        if self.D_data.shape[1] == 1:
            # Repeat single channel to create 3-channel RGB-like format
            self.D_data = self.D_data.repeat(1, 3, 1, 1)
            self.u_data = self.u_data.repeat(1, 3, 1, 1)
            print(f"  Expanded to 3 channels for VAE compatibility")
        elif self.D_data.shape[1] != 3:
            # If not 1 or 3 channels, take first 3 or pad
            if self.D_data.shape[1] > 3:
                self.D_data = self.D_data[:, :3, :, :]
                self.u_data = self.u_data[:, :3, :, :]
                print(f"  Truncated to first 3 channels")
            else:
                # Pad with zeros to reach 3 channels
                pad_channels = 3 - self.D_data.shape[1]
                self.D_data = torch.cat([
                    self.D_data,
                    torch.zeros(self.D_data.shape[0], pad_channels, *self.D_data.shape[2:])
                ], dim=1)
                self.u_data = torch.cat([
                    self.u_data,
                    torch.zeros(self.u_data.shape[0], pad_channels, *self.u_data.shape[2:])
                ], dim=1)
                print(f"  Padded to 3 channels")
        
        # Final check
        print(f"  Final D shape: {self.D_data.shape}")
        print(f"  Final u shape: {self.u_data.shape}")
        assert self.D_data.shape[1] == 3, f"D must have 3 channels, got {self.D_data.shape[1]}"
        assert self.u_data.shape[1] == 3, f"u must have 3 channels, got {self.u_data.shape[1]}"
        
    def __len__(self):
        return len(self.D_data)
    
    def __getitem__(self, idx):
        D = self.D_data[idx]  # Target: diffusion coefficient (full resolution)
        u = self.u_data[idx]  # Condition: solution field (to be downsampled)
        
        # Resize to target size if needed
        if D.shape[-1] != self.target_size or D.shape[-2] != self.target_size:
            D = TF.resize(D, (self.target_size, self.target_size), antialias=True)
            u = TF.resize(u, (self.target_size, self.target_size), antialias=True)
        
        # Ensure values are in [0, 1] range
        D = torch.clamp(D, 0, 1)
        u = torch.clamp(u, 0, 1)
        
        # Average u to create 16x16 patch representation for conditioning
        u_patches = image_to_average(u, target_size=(self.condition_patch_size, self.condition_patch_size))
        
        # Scale to [-1, 1] for VAE (common for image models)
        D = 2 * D - 1
        u_condition = 2 * u_patches - 1
        
        return D, u_condition


def build_pde_dataset(
    train_path: str,
    val_path: Optional[str] = None,
    target_size: int = 256,
    normalize: bool = True,
    condition_patch_size: int = 16,
) -> Tuple[PDEDataset, Optional[PDEDataset]]:
    """Build train and validation PDE datasets
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data (optional)
        target_size: Target image size
        normalize: Whether to normalize data
        condition_patch_size: Size of condition patches
    
    Returns:
        train_dataset, val_dataset (None if val_path not provided)
    """
    train_dataset = PDEDataset(train_path, target_size, normalize, condition_patch_size)
    
    if val_path:
        val_dataset = PDEDataset(val_path, target_size, normalize, condition_patch_size)
    else:
        val_dataset = None
    
    return train_dataset, val_dataset


# Test the dataset if run directly
if __name__ == "__main__":
    # Test loading
    train_path = "/home/ys460/Desktop/Inverse_Problem/VQVAE_generation/DiffusionCoefficient_Train1000/grf_data_1000.npz"
    val_path = "/home/ys460/Desktop/Inverse_Problem/VQVAE_generation/DiffusionCoefficient_Test100/grf_data_100.npz"
    
    try:
        train_dataset, val_dataset = build_pde_dataset(train_path, val_path)
        
        print(f"\nTrain dataset size: {len(train_dataset)}")
        print(f"Val dataset size: {len(val_dataset) if val_dataset else 'N/A'}")
        
        # Test getting a sample
        D, u_condition = train_dataset[0]
        print(f"\nSample shapes:")
        print(f"  D shape: {D.shape}, range: [{D.min():.3f}, {D.max():.3f}]")
        print(f"  u_condition shape: {u_condition.shape}, range: [{u_condition.min():.3f}, {u_condition.max():.3f}]")
        
        # Check that output is 3-channel
        assert D.shape[0] == 3, f"D must have 3 channels, got {D.shape[0]}"
        assert u_condition.shape[0] == 3, f"u_condition must have 3 channels, got {u_condition.shape[0]}"
        
        # Verify that u_condition has the patch structure (should be blocky when upsampled)
        print("\nu_condition represents 16x16 averaged patches upsampled to 256x256")
        
        # Test the averaging function directly
        u_original = train_dataset.u_data[0]  # Original u before processing
        u_patches = image_to_average(u_original, target_size=(16, 16))
        print(f"  Original u shape: {u_original.shape}")
        print(f"  Averaged patches shape: {u_patches.shape}")
        
        print("\nDataset test passed!")
    except Exception as e:
        print(f"Dataset test failed: {e}")
        import traceback
        traceback.print_exc()