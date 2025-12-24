from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import distributed as tdist, nn as nn
from torch.nn import functional as F

import dist


# this file only provides the VectorQuantizer2 used in VQVAE
__all__ = ['VectorQuantizer2',]


class VectorQuantizer2(nn.Module):
    # VQGAN originally use beta=1.0, never tried 0.25; SD seems using 0.25
    def __init__(
        self, vocab_size, Cvae, using_znorm, beta: float = 0.25,
        default_qresi_counts=0, v_patch_nums=None, quant_resi=0.5, share_quant_resi=4,  # share_quant_resi: args.qsr
    ):
        super().__init__()
        self.vocab_size: int = vocab_size
        self.Cvae: int = Cvae
        self.using_znorm: bool = using_znorm
        self.v_patch_nums: Tuple[int] = v_patch_nums
        
        self.quant_resi_ratio = quant_resi
        if share_quant_resi == 0:   # non-shared: \phi_{1 to K} for K scales
            self.quant_resi = PhiNonShared([(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(default_qresi_counts or len(self.v_patch_nums))])
        elif share_quant_resi == 1: # fully shared: only a single \phi for K scales
            self.quant_resi = PhiShared(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
        else:                       # partially shared: \phi_{1 to share_quant_resi} for K scales
            self.quant_resi = PhiPartiallyShared(nn.ModuleList([(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(share_quant_resi)]))
        
        self.register_buffer('ema_vocab_hit_SV', torch.full((len(self.v_patch_nums), self.vocab_size), fill_value=0.0))
        self.record_hit = 0
        
        self.beta: float = beta
        self.embedding = nn.Embedding(self.vocab_size, self.Cvae)
        
        # only used for progressive training of VAR (not supported yet, will be tested and supported in the future)
        self.prog_si = -1   # progressive training: not supported yet, prog_si always -1
    
    def eini(self, eini):
        if eini > 0: nn.init.trunc_normal_(self.embedding.weight.data, std=eini)
        elif eini < 0: self.embedding.weight.data.uniform_(-abs(eini) / self.vocab_size, abs(eini) / self.vocab_size)
    
    def extra_repr(self) -> str:
        return f'{self.v_patch_nums}, znorm={self.using_znorm}, beta={self.beta}  |  S={len(self.v_patch_nums)}, quant_resi={self.quant_resi_ratio}'
    
    # ===================== `forward` is only used in VAE training =====================
    def forward(self, f_BChw: torch.Tensor, ret_usages=False) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
        dtype = f_BChw.dtype
        if dtype != torch.float32: f_BChw = f_BChw.float()
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)
        
        with torch.cuda.amp.autocast(enabled=False):
            mean_vq_loss: torch.Tensor = 0.0
            vocab_hit_V = torch.zeros(self.vocab_size, dtype=torch.float, device=f_BChw.device)
            SN = len(self.v_patch_nums)
            for si, pn in enumerate(self.v_patch_nums): # from small to large
                # find the nearest embedding
                if self.using_znorm:
                    rest_NC = F.interpolate(f_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                    rest_NC = F.normalize(rest_NC, dim=-1)
                    idx_N = torch.argmax(rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
                else:
                    rest_NC = F.interpolate(f_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                    d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                    d_no_grad.addmm_(rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
                    idx_N = torch.argmin(d_no_grad, dim=1)
                
                hit_V = idx_N.bincount(minlength=self.vocab_size).float()
                if self.training:
                    if dist.initialized(): handler = tdist.all_reduce(hit_V, async_op=True)
                
                # calc loss
                idx_Bhw = idx_N.view(B, pn, pn)
                h_BChw = F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (si != SN-1) else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
                h_BChw = self.quant_resi[si/(SN-1)](h_BChw)
                f_hat = f_hat + h_BChw
                f_rest -= h_BChw
                
                if self.training and dist.initialized():
                    handler.wait()
                    if self.record_hit == 0: self.ema_vocab_hit_SV[si].copy_(hit_V)
                    elif self.record_hit < 100: self.ema_vocab_hit_SV[si].mul_(0.9).add_(hit_V.mul(0.1))
                    else: self.ema_vocab_hit_SV[si].mul_(0.99).add_(hit_V.mul(0.01))
                    self.record_hit += 1
                vocab_hit_V.add_(hit_V)
                mean_vq_loss += F.mse_loss(f_hat.data, f_BChw).mul_(self.beta) + F.mse_loss(f_hat, f_no_grad)
            
            mean_vq_loss *= 1. / SN
            f_hat = (f_hat.data - f_no_grad).add_(f_BChw)
        
        margin = tdist.get_world_size() * (f_BChw.numel() / f_BChw.shape[1]) / self.vocab_size * 0.08
        # margin = pn*pn / 100
        if ret_usages: usages = [(self.ema_vocab_hit_SV[si] >= margin).float().mean().item() * 100 for si, pn in enumerate(self.v_patch_nums)]
        else: usages = None
        return f_hat, usages, mean_vq_loss
    # ===================== `forward` is only used in VAE training =====================
    
    '''
    Algorithm 2: Multi-Scale VQVAE Reconstruction
    '''
    def embed_to_fhat(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale=True, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        ls_f_hat_BChw = []
        B = ms_h_BChw[0].shape[0]
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        if all_to_max_scale:
            # Step 1. Initial f_hat = 0
            f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, H, W, dtype=torch.float32)

            # Step 2. Accumulate Embeddings Across Scales
            for si, pn in enumerate(self.v_patch_nums): # from small to large

                # Step 2a. r_k = queue_pop(R), z_k = lookup(Z, r_k)
                h_BChw = ms_h_BChw[si]

                # Step 2b. Upsample to Full Resolution z_k = interpolate(z_k, size=(H, W))
                if si < len(self.v_patch_nums) - 1:
                    h_BChw = F.interpolate(h_BChw, size=(H, W), mode='bicubic')

                # Step 2c. f_hat = f_hat + phi_k(z_k)
                h_BChw = self.quant_resi[si/(SN-1)](h_BChw)
                f_hat.add_(h_BChw)

                if last_one: ls_f_hat_BChw = f_hat
                else: ls_f_hat_BChw.append(f_hat.clone())
                
        else:
            # WARNING: this is not the case in VQ-VAE training or inference (we'll interpolate every token map to the max H W, like above)
            # WARNING: this should only be used for experimental purpose
            f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, self.v_patch_nums[0], self.v_patch_nums[0], dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums): # from small to large
                f_hat = F.interpolate(f_hat, size=(pn, pn), mode='bicubic')
                h_BChw = self.quant_resi[si/(SN-1)](ms_h_BChw[si])
                f_hat.add_(h_BChw)
                if last_one: ls_f_hat_BChw = f_hat
                else: ls_f_hat_BChw.append(f_hat)
        
        return ls_f_hat_BChw
    
    '''
    Algorithm 1: Multi-Scale VQVAE Encoding
    '''
    def f_to_idxBl_or_fhat(self, f_BChw: torch.Tensor, to_fhat: bool, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[Union[torch.Tensor, torch.LongTensor]]:  # z_BChw is the feature from inp_img_no_grad
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()

        # Step 1. Initial Feature Extraction
        ## f = e(im), R = []
        f_rest = f_no_grad.clone() # Residual to be encoded
        f_hat = torch.zeros_like(f_rest) # Accumulated reconstruction
        f_hat_or_idx_Bl: List[torch.Tensor] = []
        
        patch_hws = [(pn, pn) if isinstance(pn, int) else (pn[0], pn[1]) for pn in (v_patch_nums or self.v_patch_nums)]    # from small to large
        assert patch_hws[-1][0] == H and patch_hws[-1][1] == W, f'{patch_hws[-1]=} != ({H=}, {W=})'
        
        SN = len(patch_hws)
        for si, (ph, pw) in enumerate(patch_hws): # from small to large
            if 0 <= self.prog_si < si: break    # progressive training: not supported yet, prog_si always -1

            # Step 2a. r_k = Q(interpolate(f_rest, size=(ph, pw)))
            ## Downsamples the residual feature map to the current patch resolution
            ## Use areas interpolation (averaging) 
            z_NC = F.interpolate(f_rest, size=(ph, pw), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)

            ## Q(): quantization operation 
            # Step 2b. queue_push() : find the nearest embedding
            if self.using_znorm:
                z_NC = F.normalize(z_NC, dim=-1)
                idx_N = torch.argmax(z_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
            else:
                d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                d_no_grad.addmm_(z_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
                idx_N = torch.argmin(d_no_grad, dim=1)
            
            # Step 2c. lookup(Z, r_k): lookup embeddings
            idx_Bhw = idx_N.view(B, ph, pw)

            # Step 2d. Upsample to Full Resolution z_k = interpolate(z_k, size=(H, W))
            h_BChw = F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (si != SN-1) else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()

            # Step 2e. f = f - phi_k(z_k); R = R + {z_k}
            h_BChw = self.quant_resi[si/(SN-1)](h_BChw) # Apply phi_k(quant_resi) A residual convolution that refines the quantized output
            f_hat.add_(h_BChw)
            f_rest.sub_(h_BChw)
            f_hat_or_idx_Bl.append(f_hat.clone() if to_fhat else idx_N.reshape(B, ph*pw))
        
        return f_hat_or_idx_Bl
    

    '''
    Algorithm 1 Modified: Multi-Scale VQVAE Encoding
    '''
    def f_to_idxBl_or_fhat_with_residuals(
        self, 
        f_BChw: torch.Tensor, 
        v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
        used_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None
    ) -> Tuple[List[torch.LongTensor], List[torch.Tensor]]:
        """
        Encode with accumulated residuals between saved scales.
        Returns: (token_list, residual_list)
        """
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)
        
        token_list: List[torch.Tensor] = []
        residual_list: List[torch.Tensor] = []
        
        # All scales to process
        all_patch_hws = [(pn, pn) if isinstance(pn, int) else (pn[0], pn[1]) 
                        for pn in (v_patch_nums or self.v_patch_nums)]
        
        # Convert used_patch_nums to set
        if used_patch_nums is not None:
            used_patch_set = set()
            for pn in used_patch_nums:
                if isinstance(pn, int):
                    used_patch_set.add((pn, pn))
                else:
                    used_patch_set.add((pn[0], pn[1]))
        else:
            used_patch_set = set(all_patch_hws)
        
        assert all_patch_hws[-1][0] == H and all_patch_hws[-1][1] == W
        
        SN = len(all_patch_hws)
        
        # Accumulators for intermediate scales
        h_total_add = torch.zeros_like(f_rest)
        
        for si, (ph, pw) in enumerate(all_patch_hws):
            if 0 <= self.prog_si < si: 
                break
            
            is_used_scale = (ph, pw) in used_patch_set
            
            # ===== ALWAYS QUANTIZE (for all scales) =====
            z_NC = F.interpolate(f_rest, size=(ph, pw), mode='area').permute(0, 2, 3, 1).reshape(-1, C) \
                if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
            
            # Q(): Find nearest embedding
            if self.using_znorm:
                z_NC = F.normalize(z_NC, dim=-1)
                idx_N = torch.argmax(z_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
            else:
                d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + \
                        torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                d_no_grad.addmm_(z_NC, self.embedding.weight.data.T, alpha=-2, beta=1)
                idx_N = torch.argmin(d_no_grad, dim=1)
            
            # lookup(): Get embeddings
            idx_Bhw = idx_N.view(B, ph, pw)
            h_BChw = F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (si != SN-1) else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
            
            # Apply phi_si
            h_BChw = self.quant_resi[si/(SN-1)](h_BChw)
            
            # Accumulate
            h_total_add = h_total_add + h_BChw
            f_hat.add_(h_BChw)
            f_rest.sub_(h_BChw)
            
            if is_used_scale:
                # ===== SAVE TOKEN and APPLY ACCUMULATED RESIDUALS =====
                token_list.append(idx_N.reshape(B, ph*pw))
                
                # Store the accumulated residual for reconstruction
                residual_list.append(h_total_add.clone())
                
                # Reset accumulator for next group
                h_total_add = torch.zeros_like(f_rest)
        
        return token_list, residual_list

    '''
    Algorithm 2 Modified: Multi-Scale VQVAE Reconstruction with Residuals
    '''
    def embed_to_fhat_with_residuals(
        self, 
        ms_h_BChw: List[torch.Tensor],
        residual_list: List[torch.Tensor],
        all_to_max_scale=True, 
        last_one=False,
        used_patch_nums=None
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Reconstruct using saved tokens and accumulated residuals.
        """
        ls_f_hat_BChw = []
        B = ms_h_BChw[0].shape[0]
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        
        # Convert to set
        f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, H, W, dtype=torch.float32)
        token_idx = 0
        residual_idx = 0
        
        for si, pn in enumerate(used_patch_nums):
            # Add the accumulated residual for this group
            f_hat = f_hat + residual_list[residual_idx]
            residual_idx += 1
            token_idx += 1
            
            # Store result
            if last_one:
                ls_f_hat_BChw = f_hat
            else:
                ls_f_hat_BChw.append(f_hat.clone())
        
        return ls_f_hat_BChw

    # ===================== idxBl_to_var_input: only used in VAR training, for getting teacher-forcing input =====================
    def idxBl_to_var_input(self, gt_ms_idx_Bl: List[torch.Tensor]) -> torch.Tensor:
        next_scales = []
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.Cvae
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        
        f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, H, W, dtype=torch.float32)
        pn_next: int = self.v_patch_nums[0]
        for si in range(SN-1):
            if self.prog_si == 0 or (0 <= self.prog_si-1 < si): break   # progressive training: not supported yet, prog_si always -1
            h_BChw = F.interpolate(self.embedding(gt_ms_idx_Bl[si]).transpose_(1, 2).view(B, C, pn_next, pn_next), size=(H, W), mode='bicubic')
            f_hat.add_(self.quant_resi[si/(SN-1)](h_BChw))
            pn_next = self.v_patch_nums[si+1]
            next_scales.append(F.interpolate(f_hat, size=(pn_next, pn_next), mode='area').view(B, C, -1).transpose(1, 2))
        return torch.cat(next_scales, dim=1) if len(next_scales) else None    # cat BlCs to BLC, this should be float32
    
    # ===================== get_next_autoregressive_input: only used in VAR inference, for getting next step's input =====================
    def get_next_autoregressive_input(self, si: int, SN: int, f_hat: torch.Tensor, h_BChw: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]: # only used in VAR inference
        HW = self.v_patch_nums[-1]
        if si != SN-1:
            h = self.quant_resi[si/(SN-1)](F.interpolate(h_BChw, size=(HW, HW), mode='bicubic'))     # conv after upsample
            f_hat.add_(h)
            return f_hat, F.interpolate(f_hat, size=(self.v_patch_nums[si+1], self.v_patch_nums[si+1]), mode='area')
        else:
            h = self.quant_resi[si/(SN-1)](h_BChw)
            f_hat.add_(h)
            return f_hat, f_hat


    # ===================== Modified for VAR training with residuals =====================
    def idxBl_to_var_input_with_residuals(
        self, 
        gt_ms_idx_Bl: List[torch.Tensor],
        gt_residual_list: List[torch.Tensor],
        used_patch_nums: Optional[Sequence[int]] = None
    ) -> torch.Tensor:
        """
        Get teacher-forcing input for VAR training with residuals.
        
        Args:
            gt_ms_idx_Bl: Ground truth tokens at used scales [r_1, r_4, r_8, r_16]
            gt_residual_list: Ground truth residuals [H_1, H_2, H_3, H_4]
            used_patch_nums: (1, 4, 8, 16)
        
        Returns:
            Concatenated inputs for next scale predictions
        """
        next_scales = []
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.Cvae
        H = W = self.v_patch_nums[-1]
        
        if used_patch_nums is None:
            used_patch_nums = self.v_patch_nums
        
        used_patch_list = [pn if isinstance(pn, int) else pn[0] for pn in used_patch_nums]
        SN = len(used_patch_list)
        
        # Build f_hat by accumulating residuals
        f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, H, W, dtype=torch.float32)
        
        for si in range(SN - 1):  # Predict up to second-to-last scale
            # Add the residual for the current scale
            f_hat = f_hat + gt_residual_list[si]
            
            # Downsample to next scale's resolution for VAR input
            pn_next = used_patch_list[si + 1]
            next_scale_input = F.interpolate(f_hat, size=(pn_next, pn_next), mode='area')
            next_scale_input = next_scale_input.view(B, C, -1).transpose(1, 2)  # B, L, C
            next_scales.append(next_scale_input)
        
        return torch.cat(next_scales, dim=1) if len(next_scales) else None


class Phi(nn.Conv2d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks//2)
        self.resi_ratio = abs(quant_resi)
    
    def forward(self, h_BChw):
        return h_BChw.mul(1-self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)


class PhiShared(nn.Module):
    def __init__(self, qresi: Phi):
        super().__init__()
        self.qresi: Phi = qresi
    
    def __getitem__(self, _) -> Phi:
        return self.qresi


class PhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]
    
    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'


class PhiNonShared(nn.ModuleList):
    def __init__(self, qresi: List):
        super().__init__(qresi)
        # self.qresi = qresi
        K = len(qresi)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())
    
    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'
