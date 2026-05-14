import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import repeat
import collections.abc
from torch.nn.init import _calculate_fan_in_and_fan_out
import torch.utils.checkpoint as checkpoint
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from itertools import repeat
import timm
import numpy as np
from functools import partial, reduce
from operator import mul
from timm.layers.config import set_reentrant_ckpt
from timm.layers import LayerNorm, RmsNorm
import torch.utils.checkpoint as checkpoint
set_reentrant_ckpt(False)
torch.backends.cuda.enable_flash_sdp(True)
from torch import nn
from .aasist import GraphAttentionLayer, HtrgGraphAttentionLayer, GraphPool, Residual_block

def interpolate(x, ratio):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.

    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled

# from PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

# --------------------------------------------------------
# 2D sine-cosine position embedding
# --------------------------------------------------------
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)  # FIX: np.float -> np.float32
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim], dtype=np.float32), pos_embed], axis=0)  # FIX: dtype
    return pos_embed

def get_2d_sincos_pos_embed_flexible(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim], dtype=np.float32), pos_embed], axis=0)  # FIX: dtype
    return pos_embed

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=(512, 128), patch_size=16, in_chans=1, embed_dim=1024, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_hw = (img_size[1] // patch_size[1], img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.init_weights()
        
    def init_weights(self):
        nn.init.trunc_normal_(self.proj.weight, std=0.01)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x
    
class PatchEmbed2D(nn.Module):
    """
    Mel-spectrogram patch embedding:
    e.g. img_size=(512, 128) 32 * 8 patches 32 * 6 = 192 patches
    """
    def __init__(self, img_size=(512, 128), patch_size=(16, 16), in_chans=1, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        assert img_size[0] % patch_size[0] == 0, "Time dimension must be divisible by patch height"
        assert img_size[1] % patch_size[1] == 0, "Frequency dimension must be divisible by patch width"

        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.patch_hw = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.init_weights()

    def init_weights(self):
        # ViT-style initialization tuned for Mel-spectrograms
        nn.init.trunc_normal_(self.proj.weight, std=0.01)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        # x: (B, C, T, N)
        B, C, T, N = x.shape
        x = self.proj(x)                     # (B, embed_dim, T/patch_h, N/patch_w)
        x = x.flatten(2).transpose(1, 2)     # (B, num_patches, embed_dim)
        x = self.norm(x)
        return x
    
def override_norm_eps(model):
    """Recursively set eps for all LayerNorm / RMSNorm layers."""
    for module in model.modules():
        if isinstance(module, (nn.LayerNorm, LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, RmsNorm)):
            module.eps = 1.e-6
    print(f"✅ All norm layers updated to eps={1.e-6}")

class AudioMosaicTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, 
            # Encoder
            spec_size=(1024, 128), patch_size=(16, 16),
            in_chans=1, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4., qkv_bias=True, qk_norm=False, proj_bias=False, pre_norm=True,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0, use_checkpoint=False, 
            norm_layer_type='layernorm', act_type='gelu',  mask_ratio=None, pos_trainable=False,
            mask_ratio_time=0.8, mask_ratio_freq=0.8,
            # Training tricks
            checkpointing_fraction=1.0, freeze_patch_embed=False, debug=False,
        ):
        if norm_layer_type == 'layernorm':
            norm_layer = LayerNorm
        elif norm_layer_type == "rmsnorm":
            norm_layer = RmsNorm
        if act_type == 'gelu':
            act_layer = nn.GELU
        elif act_type == 'silu':
            act_layer = nn.SiLU
        super(AudioMosaicTransformer, self).__init__(
            img_size=spec_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, pre_norm=pre_norm,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm, proj_bias=proj_bias,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_layer=norm_layer, act_layer=act_layer,
        )
        self.img_size = spec_size
        self.grad_checkpointing = use_checkpoint
        self.checkpointing_fraction = checkpointing_fraction
        self.patch_embed = PatchEmbed(
            img_size=spec_size, patch_size=patch_size, in_chans=1, embed_dim=embed_dim, norm_layer=norm_layer
        )
        self.norm = norm_layer(self.num_features)
        self.mask_ratio = mask_ratio
        self.mask_ratio_time = mask_ratio_time
        self.mask_ratio_freq = mask_ratio_freq
        self.pos_trainable = pos_trainable
        #self.split_pos = split_pos # not useful
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=pos_trainable)  # fixed sin-cos embedding
        self.debug = debug
        self._init_weights()
        override_norm_eps(self)
        if freeze_patch_embed:
            for param in self.patch_embed.parameters():
                param.requires_grad = False

    def _init_weights(self):
        for m in self.modules():
            # Linear and Conv2d layers
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            # Normalization layers
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm1d, RmsNorm, LayerNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

        if hasattr(self, "pos_embed") and self.pos_embed is not None:
            # pos embeds
            pos_embed = get_2d_sincos_pos_embed_flexible(self.pos_embed.shape[-1], self.patch_embed.patch_hw, cls_token=True)       
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore, ids_keep 

    def time_masking(self, x, mask_ratio):
        # x: [B, T, F, D] or [B, L, D] reshaped to [B, T, F, D]
        B, T, F, D = x.shape
        len_keep = int(T * (1 - mask_ratio))
        noise = torch.rand(B, T, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        # keep unmasked time steps
        x_masked = torch.gather(x, dim=1, index=ids_keep[:, :, None, None].expand(-1, -1, F, D))
        x_masked = x_masked.reshape(B, -1, D)

        mask_time = torch.ones([B, T], device=x.device)
        mask_time[:, :len_keep] = 0
        mask_time = torch.gather(mask_time, dim=1, index=ids_restore)
        mask = mask_time.unsqueeze(-1).expand(-1, -1, F).reshape(B, -1)

        freq_idx = torch.arange(F, device=x.device)
        ids_keep_tokens = (ids_keep.unsqueeze(-1) * F + freq_idx).reshape(B, -1)
        ids_restore_tokens = (ids_restore.unsqueeze(-1) * F + freq_idx).reshape(B, -1)

        return x_masked, mask, ids_restore_tokens, ids_keep_tokens    
    
    def freq_masking(self, x, mask_ratio):
        # x: [B, T, F, D]
        B, T, F, D = x.shape
        len_keep = int(F * (1 - mask_ratio))
        noise = torch.rand(B, F, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        # keep unmasked frequency bands
        x_masked = torch.gather(x, dim=2, index=ids_keep[:, None, :, None].expand(-1, T, -1, D))
        x_masked = x_masked.reshape(B, -1, D)

        mask_freq = torch.ones([B, F], device=x.device)
        mask_freq[:, :len_keep] = 0
        mask_freq = torch.gather(mask_freq, dim=1, index=ids_restore)
        mask = mask_freq.unsqueeze(1).expand(-1, T, -1).reshape(B, -1)

        time_idx = torch.arange(T, device=x.device).view(1, T, 1)
        ids_keep_tokens = (time_idx * F + ids_keep.view(B, 1, -1)).reshape(B, -1)
        ids_restore_tokens = (time_idx * F + ids_restore.view(B, 1, -1)).reshape(B, -1)

        return x_masked, mask, ids_restore_tokens, ids_keep_tokens
  
    def forward_features(self, x: torch.Tensor, get_layer_results=False, get_mask=False, masking_type=None) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        mask, ids_restore, ids_keep = None, None, None
        B, L, D = x.shape
        if self.training and (self.mask_ratio is not None or masking_type is not None):
            cls, x = x[:, :1, :], x[:, 1:, :]
            if masking_type == 'time':
                T = self.patch_embed.patch_hw[0]
                F = self.patch_embed.patch_hw[1]
                x = x.view(B, T, F, D)
                x, mask, ids_restore, ids_keep = self.time_masking(x, self.mask_ratio_time)
            elif masking_type == 'freq':
                T = self.patch_embed.patch_hw[0]
                F = self.patch_embed.patch_hw[1]
                x = x.view(B, T, F, D)
                x, mask, ids_restore, ids_keep = self.freq_masking(x, self.mask_ratio_freq)
            else:
                x, mask, ids_restore, ids_keep = self.random_masking(x, self.mask_ratio)
            x = torch.cat((cls, x), dim=1)

        layer_results = []
        if self.grad_checkpointing and not torch.jit.is_scripting():
            layer_index = int(len(self.blocks) * self.checkpointing_fraction)
            for i, blk in enumerate(self.blocks):
                if i < layer_index:
                    x = checkpoint.checkpoint(blk, x, use_reentrant=False)
                else:
                    x = blk(x)
                if get_layer_results:
                    layer_results.append(x)
        else:
            for blk in self.blocks:
                x = blk(x)
                if get_layer_results:
                    layer_results.append(x)
        x = self.norm(x)
        if get_layer_results:
            return x, layer_results
        if get_mask:
            return x, mask, ids_restore, ids_keep
        return x
    
    def forward(self, x, get_layer_results=False, masking_type=None): # out_feat_keys: List[str] = None):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = self.forward_features(x, get_layer_results=get_layer_results, masking_type=masking_type)
        if get_layer_results:
            x, layer_results = x
            return x, layer_results
        return x
    
class BatchNorm1dNoBias(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias.requires_grad = False

class AudioMosaicPretrain(AudioMosaicTransformer):
    def __init__(self, 
             # Encoder
            spec_size=(1024, 128), patch_size=(16, 16),
            in_chans=1, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4., qkv_bias=True, qk_norm=False, proj_bias=False, pre_norm=True,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0, use_checkpoint=False, 
            norm_layer_type='layernorm', act_type='gelu', mask_ratio=None, pos_trainable=False, 
            mask_ratio_time=0.8, mask_ratio_freq=0.8,
            # Training tricks
            checkpointing_fraction=1.0, freeze_patch_embed=False,
            # Contrastive head
            logit_scale_init=0.07, pooling='cls', learnable_tau=False,
            project_embedding_dim=128, learnable_bias=False, online_classifier=False,
            num_classes=527, use_fc_norm=False,
        **kwargs):
        super(AudioMosaicPretrain, self).__init__(
            # Encoder
            spec_size=spec_size, patch_size=patch_size, pre_norm=pre_norm,
            in_chans=in_chans, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm, proj_bias=proj_bias,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, use_checkpoint=use_checkpoint,
            norm_layer_type=norm_layer_type, act_type=act_type, mask_ratio=mask_ratio, pos_trainable=pos_trainable, 
            mask_ratio_time=mask_ratio_time, mask_ratio_freq=mask_ratio_freq,
            # Training tricks
            checkpointing_fraction=checkpointing_fraction, freeze_patch_embed=freeze_patch_embed,
        )
        del self.head
        self.checkpointing_fraction = checkpointing_fraction
        self.pooling = pooling
        self.learnable_tau = learnable_tau
        self.learnable_bias = learnable_bias
        self.online_classifier = online_classifier
        self.use_fc_norm = use_fc_norm
        if learnable_tau:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / logit_scale_init))
        if learnable_bias:
            self.logit_bias = nn.Parameter(torch.zeros([]))
        if online_classifier:
            self.online_classifier = nn.Linear(embed_dim, num_classes)
            self.online_classifier.weight.data.normal_(mean=0.0, std=0.01)
            self.online_classifier.bias.data.zero_()
        if use_fc_norm:
            self.norm = nn.Identity()
            if norm_layer_type == 'layernorm':
                self.fc_norm = LayerNorm(embed_dim)
                nn.init.constant_(self.fc_norm.bias, 0)
                nn.init.constant_(self.fc_norm.weight, 1.0)
            elif norm_layer_type == "rmsnorm":
                self.fc_norm = RmsNorm(embed_dim)
                nn.init.constant_(self.fc_norm.weight, 1.0)
        else:
            self.fc_norm = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=False),
            nn.Linear(512, project_embedding_dim, bias=False),
            BatchNorm1dNoBias(project_embedding_dim)
        )

        for m in self.projector.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, BatchNorm1dNoBias):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def get_features(self, x):
        x = super().forward(x)
        if self.pooling == 'avg':
            f = x.mean(dim=1) # Average Pooling
        else:
            f = x[:, 0, :] # CLS Token
        return f
    
    def get_logit_scale(self, min_tau=0.01, max_tau=1.0):
        # returns 1/tau with tau clamped to [min_tau, max_tau]
        tau = (1.0 / self.logit_scale.exp()).clamp(min_tau, max_tau)
        return 1.0 / tau

    def forward(self, x): # out_feat_keys: List[str] = None):
        if type(x) is dict:
            x, masking_type = x["x"], x["branch"]
        else:
            masking_type = None
        x = super().forward(x, masking_type=masking_type)
        toks = x
        if self.pooling == 'avg':
            f = x.mean(dim=1)
        elif self.pooling == 'cls':
            f = x[:, 0, :] # CLS Token
        z = self.projector(self.fc_norm(f))
        results= {
            "z": z,
            "f": f,
            "toks": toks,
        }
        if self.learnable_tau:
            results["logit_scale"] = self.get_logit_scale()
        if self.learnable_bias:
            results["logit_bias"] = self.logit_bias
        if self.online_classifier:
            results["online_logits"] = self.online_classifier(f.detach())
        return results
      
class AudioMosaicClassifier(AudioMosaicTransformer):
    def __init__(self, 
            # Encoder
            spec_size=(1024, 128), patch_size=(16, 16),
            in_chans=1, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4., qkv_bias=True, qk_norm=False, proj_bias=False, pre_norm=True,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0, use_checkpoint=False, 
            norm_layer_type='layernorm', act_type='gelu', pos_trainable=False, 
            # Training tricks 
            checkpointing_fraction=1.0, freeze_patch_embed=False,
            # Classifier
            num_classes=2, pooling='cls', classifier_drop_out=0.0, fusion_drop_rate=0.0,
            **kwargs
        ):
        super(AudioMosaicClassifier, self).__init__(
             # Encoder
            spec_size=spec_size, patch_size=patch_size, pre_norm=pre_norm,
            in_chans=in_chans, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm, proj_bias=proj_bias,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, use_checkpoint=use_checkpoint,
            norm_layer_type=norm_layer_type, act_type=act_type, pos_trainable=pos_trainable, 
            # Training tricks
            checkpointing_fraction=checkpointing_fraction, freeze_patch_embed=freeze_patch_embed,
        )
        self.head_drop_out = nn.Dropout(classifier_drop_out)
        self.pooling = pooling
        self.fusion_drop_rate = fusion_drop_rate
        if pooling == 'fusion':
            self.head = nn.Linear(embed_dim*2, num_classes)
        else:
            self.head = nn.Linear(embed_dim, num_classes)
        self.head.weight.data.normal_(mean=0.0, std=2.5e-5)
        self.head.bias.data.zero_()
        self.fc_norm = nn.LayerNorm(embed_dim, eps=1.e-6)
        nn.init.constant_(self.fc_norm.bias, 0)
        nn.init.constant_(self.fc_norm.weight, 1.0)
        del self.norm

    def adjust_linear_prob_train(self):
        # Set model to training mode except for fc_norm and head
        self.eval()
        self.fc_norm.train() 
        self.head.train()

    def _linear_prob_freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.head.weight.requires_grad = True
        self.head.bias.requires_grad = True
        self.fc_norm.weight.requires_grad = True
        self.fc_norm.bias.requires_grad = True
        
    def _reset_fc_norm(self):
        if isinstance(self.fc_norm, nn.LayerNorm):
            nn.init.constant_(self.fc_norm.bias, 0)
            nn.init.constant_(self.fc_norm.weight, 1.0)

    def random_masking_2d(self, x, mask_t_prob, mask_f_prob):
        """
        2D: Spectrogram (msking t and f under mask_t_prob and mask_f_prob)
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        
        N, L, D = x.shape  # batch, length, dim
        if self.img_size[0] == 1024:
            # for AS
            T=64
            F=8
        elif self.img_size[0] == 512:
            # for ESC
            T=32
            F=8
        elif self.img_size[0] == 128:
            # for SPC
            T=8
            F=8
        # mask T
        x = x.reshape(N, T, F, D)
        len_keep_T = int(T * (1 - mask_t_prob))
        noise = torch.rand(N, T, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, :len_keep_T]
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, F, D)
        #x_masked = torch.gather(x, dim=1, index=index)
        #x_masked = x_masked.reshape(N,len_keep_T*F,D)
        x = torch.gather(x, dim=1, index=index) # N, len_keep_T(T'), F, D

        # mask F
        #x = x.reshape(N, T, F, D)
        x = x.permute(0,2,1,3) # N T' F D => N F T' D
        len_keep_F = int(F * (1 - mask_f_prob))
        noise = torch.rand(N, F, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, :len_keep_F]
        #index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, T, D)
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, len_keep_T, D)
        x_masked = torch.gather(x, dim=1, index=index)
        x_masked = x_masked.permute(0,2,1,3) # N F' T' D => N T' F' D 
        #x_masked = x_masked.reshape(N,len_keep*T,D)
        x_masked = x_masked.reshape(N,len_keep_F*len_keep_T,D)
            
        return x_masked, None, None
    
    def forward_features(self, x: torch.Tensor, mask_t_prob=0.0, mask_f_prob=0.0) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        mask, ids_restore, ids_keep = None, None, None
        B, L, D = x.shape
        if self.training:
            cls, x = x[:, :1, :], x[:, 1:, :]
            x, _, _ = self.random_masking_2d(x, mask_t_prob, mask_f_prob)
            x = torch.cat((cls, x), dim=1)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            layer_index = int(len(self.blocks) * self.checkpointing_fraction)
            for i, blk in enumerate(self.blocks):
                if i < layer_index:
                    x = checkpoint.checkpoint(blk, x, use_reentrant=False)
                else:
                    x = blk(x)
        else:
            for blk in self.blocks:
                x = blk(x)
        # x = self.norm(x)
        return x
    
    def forward(self, x, mask_t_prob=0.0, mask_f_prob=0.0): # out_feat_keys: List[str] = None):
        x = self.forward_features(x, mask_t_prob, mask_f_prob)
        B, L, D = x.shape
        if self.pooling == 'avg':
            f = x.mean(dim=1) # Average Pooling
        elif self.pooling == 'fusion':
            if self.fusion_drop_rate > 0.0 and self.training:
                toks, _, _ = self.random_masking(x[:, 1:, :], self.fusion_drop_rate)
            else:
                toks = x[:, 1:, :]
            f = torch.cat([x[:, 0, :], toks.mean(dim=1)], dim=1) # CLS Token + Average Pooling
        else:
            f = x[:, 0, :] # CLS Token
        f = self.fc_norm(f)
        z = self.head(self.head_drop_out(f))
        return z
    
class AudioMosaicAASIST(AudioMosaicTransformer):
    def __init__(self, 
            # Encoder
            spec_size=(1024, 128), patch_size=(16, 16),
            in_chans=1, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4., qkv_bias=True, qk_norm=False, proj_bias=False, pre_norm=True,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0, use_checkpoint=False, 
            norm_layer_type='layernorm', act_type='gelu', pos_trainable=False, 
            # Training tricks 
            checkpointing_fraction=1.0, freeze_patch_embed=False,
            # Classifier
            num_classes=2, pooling='cls', classifier_drop_out=0.0, fusion_drop_rate=0.0,
            **kwargs
        ):
        super(AudioMosaicAASIST, self).__init__(
             # Encoder
            spec_size=spec_size, patch_size=patch_size, pre_norm=pre_norm,
            in_chans=in_chans, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm, proj_bias=proj_bias,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, use_checkpoint=use_checkpoint,
            norm_layer_type=norm_layer_type, act_type=act_type, pos_trainable=pos_trainable, 
            # Training tricks
            checkpointing_fraction=checkpointing_fraction, freeze_patch_embed=freeze_patch_embed,
        )
        # AASIST parameters
        filts = [128, [1, 32], [32, 32], [32, 64], [64, 64]]
        gat_dims = [64, 32]
        pool_ratios = [0.5, 0.5, 0.5, 0.5]
        temperatures =  [2.0, 2.0, 100.0, 100.0]

        self.LL = nn.Linear(embed_dim, 128)

        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        # RawNet2 encoder
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])))

        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1,1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1,1)),
            
        )
        # position encoding
        self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))
        
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        
        # Graph module
        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[1])
        # HS-GAL layer 
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])

        # Graph pooling layers
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        
        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

        self.fc_norm = nn.LayerNorm(embed_dim, eps=1.e-6)
        nn.init.constant_(self.fc_norm.bias, 0)
        nn.init.constant_(self.fc_norm.weight, 1.0)
        del self.norm

    def adjust_linear_prob_train(self):
        # Set model to training mode except for fc_norm and head
        self.eval()
        self.fc_norm.train() 
        self.head.train()

    def _linear_prob_freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.head.weight.requires_grad = True
        self.head.bias.requires_grad = True
        self.fc_norm.weight.requires_grad = True
        self.fc_norm.bias.requires_grad = True
        
    def _reset_fc_norm(self):
        if isinstance(self.fc_norm, nn.LayerNorm):
            nn.init.constant_(self.fc_norm.bias, 0)
            nn.init.constant_(self.fc_norm.weight, 1.0)

    def random_masking_2d(self, x, mask_t_prob, mask_f_prob):
        """
        2D: Spectrogram (msking t and f under mask_t_prob and mask_f_prob)
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        
        N, L, D = x.shape  # batch, length, dim
        if self.img_size[0] == 1024:
            # for AS
            T=64
            F=8
        elif self.img_size[0] == 512:
            # for ESC
            T=32
            F=8
        elif self.img_size[0] == 128:
            # for SPC
            T=8
            F=8
        # mask T
        x = x.reshape(N, T, F, D)
        len_keep_T = int(T * (1 - mask_t_prob))
        noise = torch.rand(N, T, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, :len_keep_T]
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, F, D)
        #x_masked = torch.gather(x, dim=1, index=index)
        #x_masked = x_masked.reshape(N,len_keep_T*F,D)
        x = torch.gather(x, dim=1, index=index) # N, len_keep_T(T'), F, D

        # mask F
        #x = x.reshape(N, T, F, D)
        x = x.permute(0,2,1,3) # N T' F D => N F T' D
        len_keep_F = int(F * (1 - mask_f_prob))
        noise = torch.rand(N, F, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, :len_keep_F]
        #index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, T, D)
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, len_keep_T, D)
        x_masked = torch.gather(x, dim=1, index=index)
        x_masked = x_masked.permute(0,2,1,3) # N F' T' D => N T' F' D 
        #x_masked = x_masked.reshape(N,len_keep*T,D)
        x_masked = x_masked.reshape(N,len_keep_F*len_keep_T,D)
            
        return x_masked, None, None
    
    def forward_features(self, x: torch.Tensor, mask_t_prob=0.0, mask_f_prob=0.0) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        mask, ids_restore, ids_keep = None, None, None
        B, L, D = x.shape
        if self.training:
            cls, x = x[:, :1, :], x[:, 1:, :]
            x, _, _ = self.random_masking_2d(x, mask_t_prob, mask_f_prob)
            x = torch.cat((cls, x), dim=1)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            layer_index = int(len(self.blocks) * self.checkpointing_fraction)
            for i, blk in enumerate(self.blocks):
                if i < layer_index:
                    x = checkpoint.checkpoint(blk, x, use_reentrant=False)
                else:
                    x = blk(x)
        else:
            for blk in self.blocks:
                x = blk(x)
        # x = self.norm(x)
        return x
    
    def forward(self, x, mask_t_prob=0.0, mask_f_prob=0.0): # out_feat_keys: List[str] = None):
        x = self.forward_features(x, mask_t_prob, mask_f_prob)
        x = self.LL(x) #(bs,frame_number,feat_out_dim)
        # post-processing on front-end features
        x = x.transpose(1, 2)   #(bs,feat_out_dim,frame_number)
        x = x.unsqueeze(dim=1) # add channel 
        x = F.max_pool2d(x, (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        # RawNet2-based encoder
        x = self.encoder(x)
        x = self.first_bn1(x)
        x = self.selu(x)
        
        w = self.attention(x)
        
        #------------SA for spectral feature-------------#
        w1 = F.softmax(w,dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_S = m.transpose(1, 2) + self.pos_S 
        
        # graph module layer
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)  # (#bs, #node, #dim)
        
        #------------SA for temporal feature-------------#
        w2 = F.softmax(w,dim=-2)
        m1 = torch.sum(x * w2, dim=-2)
     
        e_T = m1.transpose(1, 2)
       
        # graph module layer
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)
        
        # learnable master node
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        # inference 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1)

        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # inference 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        # Readout operation
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)

        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)
        
        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)
        
        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)
        
        return output

class AudioMosaicWeightedSumProbe(AudioMosaicTransformer):
    """SUPERB-style weighted-sum linear probe: learnable weight per layer, then linear head."""
    def __init__(self,
            spec_size=(1024, 128), patch_size=(16, 16),
            in_chans=1, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4., qkv_bias=True, qk_norm=False, proj_bias=False, pre_norm=True,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0, use_checkpoint=False,
            norm_layer_type='layernorm', act_type='gelu', pos_trainable=False,
            checkpointing_fraction=1.0, freeze_patch_embed=False,
            num_classes=527, pooling='avg',
            **kwargs
        ):
        super().__init__(
            spec_size=spec_size, patch_size=patch_size, pre_norm=pre_norm,
            in_chans=in_chans, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm, proj_bias=proj_bias,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            use_checkpoint=use_checkpoint, norm_layer_type=norm_layer_type, act_type=act_type,
            pos_trainable=pos_trainable, checkpointing_fraction=checkpointing_fraction,
            freeze_patch_embed=freeze_patch_embed,
        )
        self.pooling = pooling
        # Learnable per-layer weights (including embedding layer = depth+1)
        self.layer_weights = nn.Parameter(torch.zeros(depth + 1))
        self.fc_norm = nn.LayerNorm(embed_dim, eps=1.e-6)
        nn.init.constant_(self.fc_norm.bias, 0)
        nn.init.constant_(self.fc_norm.weight, 1.0)
        self.head = nn.Linear(embed_dim, num_classes)
        self.head.weight.data.normal_(mean=0.0, std=2.5e-5)
        self.head.bias.data.zero_()
        self.norm = nn.Identity()

    def _freeze_encoder(self):
        for name, param in self.named_parameters():
            if name not in ('layer_weights',) and not name.startswith('fc_norm') and not name.startswith('head'):
                param.requires_grad = False

    def adjust_linear_prob_train(self):
        self.eval()
        self.fc_norm.train()
        self.head.train()

    def forward(self, x, mask_t_prob=0.0, mask_f_prob=0.0):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x, layer_results = super().forward(x, get_layer_results=True)
        # layer_results: list of depth tensors; prepend the embedding (before blocks)
        # We use the post-norm output for the last entry (x) as the final layer
        # Stack: [embedding_output, block_0, block_1, ..., block_{depth-1}]
        # Approximate embedding layer as the input to first block
        # Actually layer_results already has all block outputs; add the normed final as extra
        all_layers = layer_results  # depth entries
        # Add embedding: re-extract from the first block's input is not trivial,
        # so we use layer_results[0..depth-1] plus the normed output
        all_layers.append(x)  # depth+1 entries total
        stacked = torch.stack(all_layers, dim=0)  # (depth+1, B, L, D)
        weights = F.softmax(self.layer_weights, dim=0)  # (depth+1,)
        weighted = (stacked * weights[:, None, None, None]).sum(dim=0)  # (B, L, D)
        if self.pooling == 'avg':
            f = weighted.mean(dim=1)
        else:
            f = weighted[:, 0, :]
        f = self.fc_norm(f)
        return self.head(f)


class AudioMosaicLayerProbe(AudioMosaicTransformer):
    """Layer-wise linear probe: probe a single specified layer independently."""
    def __init__(self,
            spec_size=(1024, 128), patch_size=(16, 16),
            in_chans=1, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4., qkv_bias=True, qk_norm=False, proj_bias=False, pre_norm=True,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0, use_checkpoint=False,
            norm_layer_type='layernorm', act_type='gelu', pos_trainable=False,
            checkpointing_fraction=1.0, freeze_patch_embed=False,
            num_classes=527, pooling='avg', probe_layer=11,
            **kwargs
        ):
        super().__init__(
            spec_size=spec_size, patch_size=patch_size, pre_norm=pre_norm,
            in_chans=in_chans, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm, proj_bias=proj_bias,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            use_checkpoint=use_checkpoint, norm_layer_type=norm_layer_type, act_type=act_type,
            pos_trainable=pos_trainable, checkpointing_fraction=checkpointing_fraction,
            freeze_patch_embed=freeze_patch_embed,
        )
        self.pooling = pooling
        self.probe_layer = probe_layer
        self.fc_norm = nn.LayerNorm(embed_dim, eps=1.e-6)
        nn.init.constant_(self.fc_norm.bias, 0)
        nn.init.constant_(self.fc_norm.weight, 1.0)
        self.head = nn.Linear(embed_dim, num_classes)
        self.head.weight.data.normal_(mean=0.0, std=2.5e-5)
        self.head.bias.data.zero_()
        self.norm = nn.Identity()

    def _freeze_encoder(self):
        for name, param in self.named_parameters():
            if not name.startswith('fc_norm') and not name.startswith('head'):
                param.requires_grad = False

    def adjust_linear_prob_train(self):
        self.eval()
        self.fc_norm.train()
        self.head.train()

    def forward(self, x, mask_t_prob=0.0, mask_f_prob=0.0):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        _, layer_results = super().forward(x, get_layer_results=True)
        # layer_results[i] is the output of block i (0-indexed)
        layer_out = layer_results[self.probe_layer]
        if self.pooling == 'avg':
            f = layer_out.mean(dim=1)
        else:
            f = layer_out[:, 0, :]
        f = self.fc_norm(f)
        return self.head(f)


class AudioMosaicAttentiveProbe(AudioMosaicTransformer):
    """Attentive probing: attention-weighted aggregation over all layers (per Rauch et al.)."""
    def __init__(self,
            spec_size=(1024, 128), patch_size=(16, 16),
            in_chans=1, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4., qkv_bias=True, qk_norm=False, proj_bias=False, pre_norm=True,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0, use_checkpoint=False,
            norm_layer_type='layernorm', act_type='gelu', pos_trainable=False,
            checkpointing_fraction=1.0, freeze_patch_embed=False,
            num_classes=527, pooling='avg', num_attn_heads=1,
            **kwargs
        ):
        super().__init__(
            spec_size=spec_size, patch_size=patch_size, pre_norm=pre_norm,
            in_chans=in_chans, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm, proj_bias=proj_bias,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            use_checkpoint=use_checkpoint, norm_layer_type=norm_layer_type, act_type=act_type,
            pos_trainable=pos_trainable, checkpointing_fraction=checkpointing_fraction,
            freeze_patch_embed=freeze_patch_embed,
        )
        self.pooling = pooling
        num_layers = depth + 1  # block outputs + final normed output
        # Multi-head attention over layers: query attends to layer representations
        self.attn_query = nn.Parameter(torch.randn(1, num_attn_heads, 1, embed_dim // num_attn_heads) * 0.02)
        self.attn_key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.num_attn_heads = num_attn_heads
        self.attn_scale = (embed_dim // num_attn_heads) ** -0.5
        # Layer norm per layer for stability
        self.layer_norms = nn.ModuleList([nn.LayerNorm(embed_dim, eps=1.e-6) for _ in range(num_layers)])
        self.fc_norm = nn.LayerNorm(embed_dim, eps=1.e-6)
        nn.init.constant_(self.fc_norm.bias, 0)
        nn.init.constant_(self.fc_norm.weight, 1.0)
        self.head = nn.Linear(embed_dim, num_classes)
        self.head.weight.data.normal_(mean=0.0, std=2.5e-5)
        self.head.bias.data.zero_()
        self.norm = nn.Identity()

    def _freeze_encoder(self):
        for name, param in self.named_parameters():
            if any(name.startswith(p) for p in ('attn_query', 'attn_key', 'attn_value', 'layer_norms', 'fc_norm', 'head')):
                continue
            param.requires_grad = False

    def adjust_linear_prob_train(self):
        self.eval()
        self.fc_norm.train()
        self.head.train()
        for ln in self.layer_norms:
            ln.train()
        # Keep attention params in train mode
        self.attn_key.train()
        self.attn_value.train()

    def forward(self, x, mask_t_prob=0.0, mask_f_prob=0.0):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x_normed, layer_results = super().forward(x, get_layer_results=True)
        # layer_results: depth block outputs; append final normed
        layer_results.append(x_normed)
        num_layers = len(layer_results)
        B = layer_results[0].shape[0]

        # Pool each layer to get per-layer representations
        pooled = []
        for i, lr in enumerate(layer_results):
            lr = self.layer_norms[i](lr)
            if self.pooling == 'avg':
                pooled.append(lr.mean(dim=1))  # (B, D)
            else:
                pooled.append(lr[:, 0, :])  # (B, D)
        stacked = torch.stack(pooled, dim=1)  # (B, num_layers, D)

        # Multi-head attention: query over layers
        H = self.num_attn_heads
        D_h = stacked.shape[-1] // H
        K = self.attn_key(stacked).view(B, num_layers, H, D_h).transpose(1, 2)  # (B, H, L, D_h)
        V = self.attn_value(stacked).view(B, num_layers, H, D_h).transpose(1, 2)  # (B, H, L, D_h)
        Q = self.attn_query.expand(B, -1, -1, -1)  # (B, H, 1, D_h)

        attn = (Q @ K.transpose(-2, -1)) * self.attn_scale  # (B, H, 1, L)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ V).squeeze(2)  # (B, H, D_h)
        out = out.reshape(B, -1)  # (B, D)

        f = self.fc_norm(out)
        return self.head(f)


def audiomosaic_pretrain(ckpt_path=None, **kwargs):
    model = AudioMosaicPretrain(**kwargs)
    if ckpt_path is not None:
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                new_state_dict[k.replace('_orig_mod.', '')] = v
            else:
                new_state_dict[k] = v
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(msg)
    return model


def interpolate_pos_embed_audio(model, checkpoint_model, orig_size, new_size):
    """Interpolate or resize positional embeddings for audio spectrogram patches.

    This function reshapes the patch embeddings into (T, F) grids, performs a
    bilinear interpolation to the new grid size, and then flattens back.
    """
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']  # [1, 1+N, C]
        embedding_size = pos_embed_checkpoint.shape[-1]
        # Only interpolate when grid sizes differ
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d" % (
                    orig_size[0], orig_size[1], new_size[0], new_size[1]
                ),
                flush=True,
            )
            # Separate cls token and patch tokens
            cls_token = pos_embed_checkpoint[:, :1, :]  # [1,1,C]
            pos_tokens = pos_embed_checkpoint[:, 1:, :]  # [1,N,C]

            # Reshape to (B, T_old, F_old, C)
            T_old, F_old = orig_size
            pos_tokens = pos_tokens.reshape(-1, T_old, F_old, embedding_size)

            # Interpolate in (T, F) with channels as C
            # (B, T, F, C) -> (B, C, T, F) for interpolation
            pos_tokens = pos_tokens.permute(0, 3, 1, 2)
            T_new, F_new = new_size
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(T_new, F_new), mode='bilinear', align_corners=False
            )
            # Back to (B, T, F, C)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1)
            # Flatten spatial dims back to sequence
            pos_tokens = pos_tokens.flatten(1, 2)
            # Concat back cls token
            new_pos_embed = torch.cat((cls_token, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


def audiomosaic_classifier(ckpt_path=None, reset_fc_norm=False, **kwargs):
    model = AudioMosaicClassifier(**kwargs)
    print(model)
    if ckpt_path is not None:
        current_model_dict = model.state_dict()
        loaded_state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        filtered_state_dict = {}
        for k, v in loaded_state_dict.items():
            if "module." in k:
                k = k.replace("module.", "")
            if "_orig_mod." in k:
                k = k.replace("_orig_mod.", "")
            if k in current_model_dict:
                if v.size() == current_model_dict[k].size():
                    filtered_state_dict[k] = v
                else:
                    print(
                        f"loading parameter: {k}, required shape: {current_model_dict[k].size()}, loaded shape: {v.size()}",
                        flush=True,
                    )
                    if "pos_embed" in k:
                        # Interpolate position embedding to match current patch grid
                        try:
                            n_patches_old = v.shape[1] - 1
                            new_h, new_w = model.patch_embed.patch_hw
                            assert (
                                n_patches_old % new_w == 0
                            ), f"Cannot infer original (T,F) from pos_embed of length {n_patches_old} with new F={new_w}"
                            orig_h = n_patches_old // new_w
                            orig_w = new_w
                            orig_size = (orig_h, orig_w)
                            new_size = (new_h, new_w)
                            temp = {"pos_embed": v}
                            interpolate_pos_embed_audio(model, temp, orig_size, new_size)
                            filtered_state_dict[k] = temp["pos_embed"]
                            print(
                                f"Interpolated {k}: ({orig_h},{orig_w}) -> ({new_h},{new_w})",
                                flush=True,
                            )
                        except Exception as e:
                            print(f"Failed to interpolate {k}: {e}", flush=True)
                            # Skip if interpolation fails
                    else:
                        # Skip keys with mismatched shapes
                        pass
            else:
                # Key not in current model; ignore
                print(f"loading parameter: {k}, not in current model", flush=True)

        msg = model.load_state_dict(filtered_state_dict, strict=False)
        print(msg, flush=True)
        if reset_fc_norm:
            model._reset_fc_norm()
            print("Reset fc_norm parameters", flush=True)
    # print(model)
    return model


def _load_pretrained_encoder(model, ckpt_path, reset_fc_norm=False):
    """Shared loader for probe models: loads pretrained weights, skipping head/probe params."""
    if ckpt_path is not None:
        current_model_dict = model.state_dict()
        loaded_state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        filtered_state_dict = {}
        for k, v in loaded_state_dict.items():
            if "module." in k:
                k = k.replace("module.", "")
            if "_orig_mod." in k:
                k = k.replace("_orig_mod.", "")
            if k in current_model_dict and v.size() == current_model_dict[k].size():
                filtered_state_dict[k] = v
            elif k in current_model_dict and "pos_embed" in k:
                try:
                    n_patches_old = v.shape[1] - 1
                    new_h, new_w = model.patch_embed.patch_hw
                    orig_h = n_patches_old // new_w
                    orig_w = new_w
                    temp = {"pos_embed": v}
                    interpolate_pos_embed_audio(model, temp, (orig_h, orig_w), (new_h, new_w))
                    filtered_state_dict[k] = temp["pos_embed"]
                except Exception as e:
                    print(f"Failed to interpolate {k}: {e}", flush=True)
        msg = model.load_state_dict(filtered_state_dict, strict=False)
        print(msg, flush=True)
    model._freeze_encoder()
    return model


def audiomosaic_weighted_sum_probe(ckpt_path=None, **kwargs):
    model = AudioMosaicWeightedSumProbe(**kwargs)
    return _load_pretrained_encoder(model, ckpt_path)


def audiomosaic_layer_probe(ckpt_path=None, **kwargs):
    model = AudioMosaicLayerProbe(**kwargs)
    return _load_pretrained_encoder(model, ckpt_path)


def audiomosaic_attentive_probe(ckpt_path=None, **kwargs):
    model = AudioMosaicAttentiveProbe(**kwargs)
    return _load_pretrained_encoder(model, ckpt_path)


def audiomosaic_aasist(ckpt_path=None, reset_fc_norm=False, **kwargs):
    model = AudioMosaicAASIST(**kwargs)
    print(model)
    if ckpt_path is not None:
        current_model_dict = model.state_dict()
        loaded_state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        filtered_state_dict = {}
        for k, v in loaded_state_dict.items():
            if "module." in k:
                k = k.replace("module.", "")
            if "_orig_mod." in k:
                k = k.replace("_orig_mod.", "")
            if k in current_model_dict:
                if v.size() == current_model_dict[k].size():
                    filtered_state_dict[k] = v
                else:
                    print(
                        f"loading parameter: {k}, required shape: {current_model_dict[k].size()}, loaded shape: {v.size()}",
                        flush=True,
                    )
                    if "pos_embed" in k:
                        # Interpolate position embedding to match current patch grid
                        try:
                            n_patches_old = v.shape[1] - 1
                            new_h, new_w = model.patch_embed.patch_hw
                            assert (
                                n_patches_old % new_w == 0
                            ), f"Cannot infer original (T,F) from pos_embed of length {n_patches_old} with new F={new_w}"
                            orig_h = n_patches_old // new_w
                            orig_w = new_w
                            orig_size = (orig_h, orig_w)
                            new_size = (new_h, new_w)
                            temp = {"pos_embed": v}
                            interpolate_pos_embed_audio(model, temp, orig_size, new_size)
                            filtered_state_dict[k] = temp["pos_embed"]
                            print(
                                f"Interpolated {k}: ({orig_h},{orig_w}) -> ({new_h},{new_w})",
                                flush=True,
                            )
                        except Exception as e:
                            print(f"Failed to interpolate {k}: {e}", flush=True)
                            # Skip if interpolation fails
                    else:
                        # Skip keys with mismatched shapes
                        pass
            else:
                # Key not in current model; ignore
                print(f"loading parameter: {k}, not in current model", flush=True)

        msg = model.load_state_dict(filtered_state_dict, strict=False)
        print(msg, flush=True)
        if reset_fc_norm:
            model._reset_fc_norm()
            print("Reset fc_norm parameters", flush=True)
    # print(model)
    return model

