import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

import dist
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2

class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class SpatialConditionEncoder(nn.Module):
    """Encoder for spatial conditioning maps"""
    def __init__(self, input_channels=1, output_dim=1024):
        super().__init__()
        # Simple linear projection - can be replaced with more complex encoders
        self.encoder = nn.Conv2d(input_channels, output_dim, kernel_size=1, bias=True)
        
    def forward(self, x):
        # x: (B, C, H, W) -> (B, D, H, W)
        return self.encoder(x)


class PDEVAR(nn.Module):
    '''VAR model with solution field U conditioning support (B, 32, 32) instead of class label (B,)'''
    def __init__(
        self, vae_local: VQVAE,
        spatial_cond_channels=32,
        spatial_cond_size=16,
        depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., 
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        self.embed_dim = embed_dim
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        
        self.patch_nums: Tuple[int] = patch_nums
        self.spatial_cond_channels = spatial_cond_channels
        self.spatial_cond_size = spatial_cond_size

        # self.L = sum(pn ** 2 for pn in self.patch_nums)
        # self.first_l = self.patch_nums[0] ** 2
        self.first_l = self.spatial_cond_size ** 2
        self.L = sum(pn ** 2 for pn in self.patch_nums[1:]) + self.first_l
        self.begin_ends = []
        cur = 0
        # First stage is spatial conditioning
        self.begin_ends.append((cur, cur + self.first_l))
        cur += self.first_l 
        # Remaining stages from patch_nums[1:]
        for i, pn in enumerate(self.patch_nums[1:]):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        
        # 1. input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)

        # Spatial conditioning encoder
        self.spatial_encoder = SpatialConditionEncoder(
            input_channels=spatial_cond_channels,
            output_dim=self.C
        )

        # Global conditioning from spatial field (similar to class embedding)
        self.spatial_global_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # （B, C, H, W) -> (B, C, 1, 1)
            nn.Flatten(), # (B, C, 1, 1) -> (B, C)
            nn.Linear(self.C, self.C) # (B, C) -> (B, C)
        )

        # Do we still need CFG for spatial conditioning?
        # Null conditioning for CFG
        self.null_cond = nn.Parameter(torch.zeros(1, self.C))
        self.null_spatial = nn.Parameter(torch.zeros(1, spatial_cond_channels, spatial_cond_size, spatial_cond_size))
        
        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        nn.init.trunc_normal_(self.null_cond.data, mean=0, std=init_std)
        nn.init.normal_(self.null_spatial.data, mean=0, std=0.02)
        
        # 3. absolute position embedding
        pos_1LC = []
        # Spatial conditioning stage
        pe_spatial = torch.empty(1, self.first_l, self.C)
        nn.init.trunc_normal_(pe_spatial, mean=0, std=init_std)
        pos_1LC.append(pe_spatial)
        # Remaining stages
        for i, pn in enumerate(self.patch_nums[1:]):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        
        pos_1LC = torch.cat(pos_1LC, dim=1)
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)

        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # 4. transformer blocks
        self.shared_ada_lin = nn.Sequential(
            nn.SiLU(inplace=False), 
            SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, 
                num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], 
                last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, 
                fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )
        
        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        # d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        lvl_indices = [0] * self.first_l # spatial conditioning
        for i, pn in enumerate(self.patch_nums[1:], start=1):
            lvl_indices.extend([i] * (pn * pn))

        d = torch.tensor(lvl_indices).view(1, self.L, 1) # d: 1L1
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)

        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)
    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    def encode_spatial(self, spatial_cond: torch.Tensor, drop: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode spatial conditioning (from u field)
        
        Args:
            spatial_cond: [B, L, C] where L=16*16=256, C=32 (from VQVAE)
            drop: whether to apply conditioning dropout
            
        Returns:
            cond_BD: class/null conditioning
            spatial_BLC: processed spatial conditioning
        """
        B, L, C_in = spatial_cond.shape
        
        # Project spatial conditioning to model dimension if needed
        if C_in != self.embed_dim:
            # Add a projection layer if not already present
            if not hasattr(self, 'spatial_proj'):
                self.spatial_proj = nn.Linear(C_in, self.embed_dim).to(spatial_cond.device)
                # Initialize the projection
                nn.init.xavier_uniform_(self.spatial_proj.weight)
                nn.init.zeros_(self.spatial_proj.bias)
            spatial_cond = self.spatial_proj(spatial_cond)  # [B, L, embed_dim]
        
        # Apply dropout if training
        if drop and self.training and self.cond_drop_rate > 0:
            # Create dropout mask
            drop_mask = torch.rand(B, 1, 1, device=spatial_cond.device) < self.cond_drop_rate
            # Apply dropout (replace with zeros for dropped samples)
            spatial_cond = torch.where(
                drop_mask.expand(B, L, self.embed_dim),
                torch.zeros_like(spatial_cond),
                spatial_cond
            )
        
        # For spatial conditioning, we don't use class conditioning
        cond_BD = torch.zeros(B, self.embed_dim, device=spatial_cond.device)
        
        return cond_BD, spatial_cond

    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, B: int, 
        spatial_cond: torch.Tensor,
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False,
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param spatial_cond: spatial conditioning tensor
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng

        # prepare conditioning with and without spatial info for CFG
        cond_BD_with, spatial_BLC_with = self.encode_spatial(spatial_cond, drop=False)
        cond_BD_null = self.null_cond.expand(B, -1)
        spatial_BLC_null = torch.zeros(B, self.first_l, self.C, device=spatial_cond.device)


        # Stack for CFG
        cond_BD = torch.cat([cond_BD_with, cond_BD_null], dim=0)  # (2B, D)
        spatial_BLC = torch.cat([spatial_BLC_with, spatial_BLC_null], dim=0)  # (2B, first_l, C)
        
        # Initialize with spatial tokens
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = spatial_BLC + self.pos_start.expand(2 * B, -1, -1) + lvl_pos[:, :self.first_l]
        
        cur_L = self.first_l  # Start after spatial conditioning
        f_hat = cond_BD.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        
        # Enable KV caching for inference
        for b in self.blocks:
            b.attn.kv_caching(True)

        for si, pn in enumerate(self.patch_nums[1:], start=1):
            ratio = si / self.num_stages_minus_1
            
            # Forward through transformer
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            
            logits_BlV = self.get_logits(x, cond_BD)
            
            # Apply CFG
            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]
            
            # Sample tokens
            idx_Bl = sample_with_top_k_top_p_(
                logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1
            )[:, :, 0]
            
            if not more_smooth:
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)
            else:
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                h_BChw = gumbel_softmax_with_rng(
                    logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng
                ) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            
            # Update feature map
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(
                si-1, len(self.patch_nums)-1, f_hat, h_BChw
            )
            
            if si != len(self.patch_nums) - 1:  # Prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)  # Double for CFG
            
            cur_L += pn * pn
        
        # Disable KV caching
        for b in self.blocks:
            b.attn.kv_caching(False)
        
        return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]
    
    def forward(self, spatial_cond: torch.Tensor, x_BLCv_wo_first_l: torch.Tensor) -> torch.Tensor:  # returns logits_BLV
        """
        :param spatial_cond: spatial conditioning tensor (B, C, H, W) or (B, H, W)
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :return: 
        """
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B = x_BLCv_wo_first_l.shape[0]

        with torch.amp.autocast(device_type='cuda', enabled=False):
            # Encode spatial conditioning
            cond_BD, spatial_BLC = self.encode_spatial(spatial_cond, drop=True)

            # Start tokens are the spatial conditioning
            sos = spatial_BLC + self.pos_start.expand(B, -1, -1) 

            if self.prog_si == 0: x_BLC = sos
            else: x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)

            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed] # lvl: BLC;  pos: 1LC
        
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        
        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)
        
        AdaLNSelfAttn.forward
        for i, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)

        x_BLC = self.get_logits(x_BLC.float(), cond_BD)
        
        # if self.prog_si == 0:
        #     if isinstance(self.word_embed, nn.Linear):
        #         x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
        #     else:
        #         s = 0
        #         for p in self.word_embed.parameters():
        #             if p.requires_grad:
        #                 s += p.view(-1)[0] * 0
        #         x_BLC[0, 0, 0] += s
        return x_BLC    # logits BLV, V is vocab_size
    
    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5     # init_std < 0: automated
        
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()
        
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'


class VARHF(PDEVAR, PyTorchModelHubMixin):
            # repo_url="https://github.com/FoundationVision/VAR",
            # tags=["image-generation"]):
    def __init__(
        self,
        vae_kwargs,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        vae_local = VQVAE(**vae_kwargs)
        super().__init__(
            vae_local=vae_local,
            num_classes=num_classes, depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_eps=norm_eps, shared_aln=shared_aln, cond_drop_rate=cond_drop_rate,
            attn_l2_norm=attn_l2_norm,
            patch_nums=patch_nums,
            flash_if_available=flash_if_available, fused_if_available=fused_if_available,
        )
