from typing import Tuple
import torch.nn as nn

from .quant import VectorQuantizer2
from .var import PDEVAR
from .vqvae import VQVAE


def build_vae_var(
    # Shared args
    device, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
    # VQVAE args
    V=4096, Cvae=32, ch=160, share_quant_resi=4,
    # PDEVAR args
    spatial_cond_channels=32, spatial_cond_size=16,  # Changed default to 32
    depth=16, shared_aln=False, attn_l2_norm=True,
    flash_if_available=True, fused_if_available=True,
    init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1,    # init_std < 0: automated
    # Additional PDEVAR parameters
    embed_dim=1024, num_heads=16, mlp_ratio=4.,
    drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
    norm_eps=1e-6, cond_drop_rate=0.1,
) -> Tuple[VQVAE, PDEVAR]:

    heads = depth
    width = depth * 64
    dpr = 0.1 * depth/24
    
    # disable built-in initialization for speed
    for clz in (nn.Linear, nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d):
        setattr(clz, 'reset_parameters', lambda self: None)
    
    # build models
    vae_local = VQVAE(vocab_size=V, z_channels=Cvae, ch=ch, test_mode=True, share_quant_resi=share_quant_resi, v_patch_nums=patch_nums).to(device)
    
    # Use the actual parameters passed in!
    var_wo_ddp = PDEVAR(
        vae_local=vae_local,
        spatial_cond_channels=spatial_cond_channels, 
        spatial_cond_size=spatial_cond_size,
        depth=depth, 
        embed_dim=embed_dim,  
        num_heads=num_heads,           
        mlp_ratio=mlp_ratio,                    
        drop_rate=drop_rate,                          
        attn_drop_rate=attn_drop_rate,               
        drop_path_rate=drop_path_rate,               
        norm_eps=norm_eps,                            
        shared_aln=shared_aln,                        
        cond_drop_rate=cond_drop_rate,               
        attn_l2_norm=attn_l2_norm,              
        patch_nums=patch_nums,                         
        flash_if_available=flash_if_available,        
        fused_if_available=fused_if_available,        
    ).to(device)
    
    var_wo_ddp.init_weights(init_adaln=init_adaln, init_adaln_gamma=init_adaln_gamma, init_head=init_head, init_std=init_std)
    
    return vae_local, var_wo_ddp
