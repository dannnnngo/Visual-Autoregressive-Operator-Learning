import math
from click import Option
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from functools import partial

# Import original VAR components
from models.basic_var import AdaLNSelfAttn, AdaLNBeforeHead

class PDEVAR(nn.Module):
    """
    Autoregressive Super-Resolution Transformer
    Predicts high-resolution image tokens progressively from a low-resolution input image.
    """
    
    def __init__(
        self,
        vqvae,  # Pretrained VQ-VAE model
        patch_nums=(1, 4, 8, 16),  # Multi-scale patch numbers
        spatial_cond_size=16,  # Size of conditioning image (16x16)
        d_model=128,  # Transformer dimension
        nhead=8,  # Number of attention heads
        num_layers=2,  # Number of transformer layers
        dim_feedforward=256,  # FFN dimension
        cond_drop_rate=0.1,  # Conditioning dropout rate for better generalization
    ):
        super().__init__()
        
        self.vqvae = vqvae
        self.patch_nums = patch_nums
        self.vocab_size = vqvae.vocab_size
        self.Cvae = vqvae.Cvae
        self.C = d_model
        self.cond_drop_rate = cond_drop_rate
        
        # Calculate sequence length
        # first_l: conditioning image tokens (16*16 = 256)
        self.first_l = spatial_cond_size * spatial_cond_size
        # Total length includes conditioning + all progressive scales
        self.L = self.first_l + sum(pn * pn for pn in self.patch_nums)
        
        print(f"Sequence length: {self.L} (conditioning: {self.first_l}, progressive: {self.L - self.first_l})")
        
        # 1. Word embedding (similar to VAR) - for continuous features from VQ-VAE
        self.word_embed = nn.Linear(self.Cvae, self.C)
        
        # Feature projection for conditioning (from spatial features to transformer dimension)
        self.feature_proj = nn.Linear(self.Cvae, self.C)
        
        # SPECIAL TOKEN for first scale (guess token)
        # This is a learnable embedding for the first scale's initial token
        init_std = math.sqrt(1 / self.C / 3)
        self.first_scale_token = nn.Parameter(torch.empty(1, self.Cvae))
        nn.init.trunc_normal_(self.first_scale_token.data, mean=0, std=init_std)
        
        # 2. Start token for conditioning
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. Absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn * pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        
        pos_1LC = torch.cat(pos_1LC, dim=1)
        self.pos_1LC = nn.Parameter(pos_1LC)
        
        # 4. Level embedding (distinguish different pyramid levels)
        # +1 because we have conditioning stage + len(patch_nums) progressive stages
        self.lvl_embed = nn.Embedding(len(self.patch_nums) + 1, self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # Create level indices for the sequence
        lvl_1L = [0] * self.first_l  # Conditioning stage is level 0
        for i, pn in enumerate(self.patch_nums, start=1):
            lvl_1L.extend([i] * (pn * pn))
        self.register_buffer('lvl_1L', torch.tensor(lvl_1L).view(1, self.L))
        
        # 5. Attention mask for causal (autoregressive) training
        d = self.lvl_1L.view(1, self.L, 1)  # Shape: 1×L×1
        dT = d.transpose(1, 2)  # Shape: 1×1×L
        
        # Causal mask: can only attend to current and previous levels
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. Transformer encoder layers
        self.cond_projection = nn.Sequential(
            nn.Linear(spatial_cond_size * spatial_cond_size, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        mlp_ratio = dim_feedforward / d_model
        
        dpr = [x.item() for x in torch.linspace(0, 0.0, num_layers)]  # stochastic depth (set to 0 for now)
        
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.C,
                shared_aln=False,
                block_idx=block_idx,
                embed_dim=self.C,
                norm_layer=norm_layer,
                num_heads=nhead,
                mlp_ratio=mlp_ratio,
                drop=0.0,
                attn_drop=0.0,
                drop_path=dpr[block_idx],
                last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=False,
                flash_if_available=True,
                fused_if_available=True,
            )
            for block_idx in range(num_layers)
        ])

        # 7. Output head to predict next tokens
        self.head_nm = AdaLNBeforeHead(self.C, self.C, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.vocab_size, bias=True)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for the model"""
        init_std = math.sqrt(1 / self.C / 3)
        
        # Initialize word embedding
        nn.init.trunc_normal_(self.word_embed.weight.data, mean=0, std=init_std)
        if self.word_embed.bias is not None:
            nn.init.zeros_(self.word_embed.bias.data)
        
        # Initialize feature projection
        nn.init.trunc_normal_(self.feature_proj.weight.data, mean=0, std=init_std)
        if self.feature_proj.bias is not None:
            nn.init.zeros_(self.feature_proj.bias.data)
        
        # Initialize output head 
        nn.init.trunc_normal_(self.head.weight.data, mean=0, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias.data)

    def forward(
        self,
        input_cond_img: Optional[torch.Tensor],
        gt_ms_idx_Bl: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        input_cond_img: (B, 16, 16) or (B, 1, 16, 16)
        gt_ms_idx_Bl: preprocessed continuous features from VQ-VAE (B, L-first_l-1, Cvae)
        """
        B = input_cond_img.shape[0]
        device = input_cond_img.device
        
        if input_cond_img.dim() == 3:
            input_cond_img = input_cond_img.unsqueeze(1)
        
        # Create global conditioning embedding for AdaLN
        cond_flat = input_cond_img.view(B, -1)  # (B, 256)
        
        # Apply conditioning dropout
        if self.training and self.cond_drop_rate > 0:
            drop_mask = torch.rand(B, 1, device=device) < self.cond_drop_rate
            cond_flat = torch.where(drop_mask, torch.zeros_like(cond_flat), cond_flat)
        
        cond_BD = self.cond_projection(cond_flat)  # (B, D)

        input_cond_flat = input_cond_img.view(B, self.first_l, 1)  # (B, 256, 1)

        # Create a learned embedding that maps single channel to Cvae
        if not hasattr(self, 'cond_channel_proj'):
            self.cond_channel_proj = nn.Linear(1, self.Cvae).to(device)
            nn.init.trunc_normal_(self.cond_channel_proj.weight, std=0.02)
        
        input_cond_cvae = self.cond_channel_proj(input_cond_flat)  # (B, 256, Cvae)
        input_cond = self.feature_proj(input_cond_cvae)  # (B, 256, C)
        
        # Add positional embedding for conditioning
        input_cond = input_cond + self.pos_start  # (B, 256, C)
        
        # Add level embedding for conditioning (level 0)
        cond_lvl_embed = self.lvl_embed(torch.zeros(B, self.first_l, dtype=torch.long, device=device))
        input_cond = input_cond + cond_lvl_embed
        
        # Process ground truth features
        # Add guess token for first scale
        guess_token = self.first_scale_token.expand(B, 1, -1)  # (B, 1, Cvae)
        
        # Concatenate guess token with ground truth
        progressive_features_cvae = torch.cat([guess_token, gt_ms_idx_Bl], dim=1)  # (B, L-first_l, Cvae)
        
        # Project to transformer dimension
        progressive_features = self.word_embed(progressive_features_cvae.float())  # (B, L-first_l, C)
        
        # Verify positional embeddings are correctly applied
        L_prog = self.L - self.first_l
        assert progressive_features.shape[1] == L_prog, f"Shape mismatch: {progressive_features.shape[1]} vs {L_prog}"
        progressive_features = progressive_features + self.pos_1LC[:, :L_prog, :].expand(B, -1, -1)
        
        # Verify level embeddings
        prog_lvl_indices = self.lvl_1L[:, self.first_l:].expand(B, -1)  # (B, L-first_l)
        
        # Add assertion to check level indices
        expected_levels = []
        for i, pn in enumerate(self.patch_nums, start=1):
            expected_levels.extend([i] * (pn * pn))
        assert len(expected_levels) == L_prog, f"Level mismatch: {len(expected_levels)} vs {L_prog}"
        
        prog_lvl_embed = self.lvl_embed(prog_lvl_indices)  # (B, L-first_l, C)
        progressive_features = progressive_features + prog_lvl_embed
        
        # Concatenate conditioning and progressive features
        x_BLC = torch.cat([input_cond, progressive_features], dim=1)  # (B, L, C)
        
        # Apply transformer with causal masking
        seq_len = x_BLC.shape[1]
        attn_bias = self.attn_bias_for_masking[:, :, :seq_len, :seq_len]
        for block in self.blocks:
            x_BLC = block(x=x_BLC, cond_BD=cond_BD, attn_bias=attn_bias)
        
        # Output head
        x_BLC = self.head_nm(x_BLC.float(), cond_BD)
        logits_BLV = self.head(x_BLC.float())  # (B, L, vocab_size)
        
        return logits_BLV

    @torch.no_grad()
    def autoregressive_generate(self, input_cond_img: Optional[torch.Tensor], condition_patch_size=16, device='cuda'):
        '''
        condition_patch_size: size of condition patches
        input_cond_img: (B, 16, 16) condition image u
        '''
        B = input_cond_img.shape[0]
        device = input_cond_img.device
        
        if input_cond_img.dim() == 3:
            input_cond_img = input_cond_img.unsqueeze(1)
        
        # Create global conditioning embedding for AdaLN (same as in forward)
        cond_flat = input_cond_img.view(B, -1)  # (B, 256)
        cond_embed = self.cond_projection(cond_flat)  # (B, D)
        
        # Process conditioning as tokens
        input_cond_flat = input_cond_img.view(B, -1, 1)  # (B, 256, 1)
        input_cond = self.feature_proj(input_cond_flat.expand(-1, -1, self.Cvae))  # (B, 256, C)
        input_cond = input_cond + self.pos_start
        cond_lvl_embed = self.lvl_embed(torch.zeros(B, self.first_l, dtype=torch.long, device=device))
        input_cond = input_cond + cond_lvl_embed
        
        # Initialize with conditioning + guess token for first scale
        # Add the guess token for the first scale
        guess_token = self.first_scale_token.expand(B, 1, -1)  # (B, 1, Cvae)
        guess_embed = self.word_embed(guess_token)  # (B, 1, C)
        guess_embed = guess_embed + self.pos_1LC[:, :1, :]  # Add position for first token
        guess_embed = guess_embed + self.lvl_embed(torch.ones(B, 1, dtype=torch.long, device=device))  # Level 1
        
        # Start with conditioning + guess token
        x_BLC = torch.cat([input_cond, guess_embed], dim=1)  # (B, first_l + 1, C)
        
        generated_tokens = []
        # Position counter for tracking position in pos_1LC
        pos_counter = 0  # Start from 0 for the progressive part
        
        # Feature map for VQ-VAE decoding
        f_hat = input_cond.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        
        # Generate tokens for each scale
        for si, pn in enumerate(self.patch_nums):
            num_tokens = pn * pn
            
            # Run transformer
            seq_len = x_BLC.shape[1]
            # Extract 2D mask (seq_len, seq_len) from 4D buffer
            mask = self.attn_bias_for_masking[0, 0, :seq_len, :seq_len]  # (seq_len, seq_len)
            
            # Apply AdaLN transformer layers with conditioning
            hidden_BLC = x_BLC
            for block in self.blocks:
                hidden_BLC = block(x=hidden_BLC, cond_BD=cond_embed, attn_bias=mask.unsqueeze(0).unsqueeze(0))
            
            hidden_BLC = self.head_nm(hidden_BLC, cond_embed)
            logits_BLV = self.head(hidden_BLC)  # (B, seq_len, vocab_size)
            
            # Sample tokens for current scale
            # We need to get the last num_tokens predictions
            # Since we've been appending, the predictions for this scale are at the end
            if si == 0:
                # First scale: predict from the guess token position (first_l)
                scale_logits = logits_BLV[:, self.first_l:self.first_l+num_tokens, :]  # (B, num_tokens, vocab_size)
            else:
                # Other scales: get the last num_tokens predictions
                scale_logits = logits_BLV[:, -num_tokens:, :]  # (B, num_tokens, vocab_size)
            
            idx_Bl = torch.argmax(scale_logits, dim=-1)  # (B, num_tokens)
            generated_tokens.append(idx_Bl)
            
            # Get embeddings from VQ-VAE codebook
            h_BChw = self.vqvae.quantize.embedding(idx_Bl)  # (B, num_tokens, Cvae)
            h_BChw = h_BChw.transpose(1, 2).reshape(B, self.Cvae, pn, pn)  # (B, Cvae, pn, pn)
            
            # Update position counter
            pos_counter += num_tokens
            
            # Update feature map
            if si < len(self.patch_nums) - 1:
                # Manually accumulate features (don't use get_next_autoregressive_input)
                # Apply phi and accumulate
                SN_vqvae = len(self.vqvae.quantize.v_patch_nums)
                h_upsampled = F.interpolate(h_BChw, size=(self.patch_nums[-1], self.patch_nums[-1]), mode='bicubic')
                h_processed = self.vqvae.quantize.quant_resi[si/(SN_vqvae-1)](h_upsampled)
                f_hat = f_hat + h_processed
                
                # Downsample to next scale for input
                pn_next = self.patch_nums[si+1]
                next_num_tokens = pn_next ** 2
                next_token_map = F.interpolate(f_hat, size=(pn_next, pn_next), mode='area')  # (B, Cvae, pn_next, pn_next)
                
                # Convert to sequence format
                next_token_map = next_token_map.reshape(B, self.Cvae, -1).transpose(1, 2)  # (B, next_num_tokens, Cvae)
                
                # Project to transformer dimension and add embeddings
                next_token_embed = self.word_embed(next_token_map)  # (B, next_num_tokens, C)
                
                # Add position embeddings - use the correct position range
                next_pos_embed = self.pos_1LC[:, pos_counter:pos_counter+next_num_tokens, :]
                next_token_embed = next_token_embed + next_pos_embed
                
                # Add level embeddings
                next_lvl = si + 2  # +1 for next scale, +1 because conditioning is level 0
                next_lvl_embed = self.lvl_embed(torch.full((B, next_num_tokens), next_lvl, dtype=torch.long, device=device))
                next_token_embed = next_token_embed + next_lvl_embed
                
                # Append to sequence
                x_BLC = torch.cat([x_BLC, next_token_embed], dim=1)
            else:
                # Last scale - just update f_hat
                SN_vqvae = len(self.vqvae.quantize.v_patch_nums)
                h_processed = self.vqvae.quantize.quant_resi[si/(SN_vqvae-1)](h_BChw)
                f_hat = f_hat + h_processed
        
        # Return both tokens and the final feature map
        return generated_tokens, f_hat  # List of (B, num_tokens) tensors, (B, Cvae, H, W)
