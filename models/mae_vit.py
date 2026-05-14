# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import numpy as np
import timm.models.vision_transformer
from timm.models.vision_transformer import PatchEmbed, Block
from timm.layers import to_2tuple
from timm.layers import trunc_normal_


class PatchEmbed_new(nn.Module):
    """ Flexible Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        
        self.img_size = img_size
        self.patch_size = patch_size
        

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride) # with overlapped patches
        #self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        #self.patch_hw = (img_size[1] // patch_size[1], img_size[0] // patch_size[0])
        #self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        _, _, h, w = self.get_output_shape(img_size) # n, emb_dim, h, w
        self.patch_hw = (h, w)
        self.num_patches = h*w

    def get_output_shape(self, img_size):
        # todo: don't be lazy..
        return self.proj(torch.randn(1,1,img_size[0],img_size[1])).shape 

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        #assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class PatchEmbed3D_new(nn.Module):
    """ Flexible Image to Patch Embedding
    """
    def __init__(self, video_size=(16,224,224), patch_size=(2,16,16), in_chans=3, embed_dim=768, stride=(2,16,16)):
        super().__init__()
        
        self.video_size = video_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        _, _, t, h, w = self.get_output_shape(video_size) # n, emb_dim, h, w
        self.patch_thw = (t, h, w)
        self.num_patches = t*h*w

    def get_output_shape(self, video_size):
        # todo: don't be lazy..
        return self.proj(torch.randn(1, self.in_chans, video_size[0], video_size[1], video_size[2])).shape 

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = self.proj(x) # 32, 3, 16, 224, 224 -> 32, 768, 8, 14, 14
        x = x.flatten(2) # 32, 768, 1568
        x = x.transpose(1, 2) # 32, 768, 1568 -> 32, 1568, 768
        return x
    

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, mask_2d=True, use_custom_patch=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.img_size = kwargs["img_size"]
        self.patch_embed = PatchEmbed_new(img_size=kwargs["img_size"], patch_size=(16,16), in_chans=1, embed_dim=kwargs["embed_dim"], stride=16) # no overlap. stride=img_size=16
        num_patches = self.patch_embed.num_patches
        #num_patches = 512 # assume audioset, 1024//16=64, 128//16=8, 512=64x8
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, kwargs["embed_dim"]), requires_grad=False)  # fixed sin-cos embedding
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = nn.LayerNorm
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
        del self.norm  # remove the original norm
        self.mask_2d = mask_2d
        self.use_custom_patch = use_custom_patch
        num_heads=12
        depth=12
        mlp_ratio=4
        trunc_normal_(self.head.weight, std=2e-5)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)        

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

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


    def forward_features_mask(self, x, mask_t_prob, mask_f_prob):
        B = x.shape[0] #4,1,1024,128
        x = self.patch_embed(x) # 4, 512, 768

        x = x + self.pos_embed[:, 1:, :]
        if self.random_masking_2d:
            x, mask, ids_restore = self.random_masking_2d(x, mask_t_prob, mask_f_prob)
        else:
            x, mask, ids_restore = self.random_masking(x, mask_t_prob)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)        
        x = self.pos_drop(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def adjust_linear_prob_train(self):
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

    # overwrite original timm
    def forward(self, x, mask_t_prob=0.0, mask_f_prob=0.0):
        if mask_t_prob > 0.0 or mask_f_prob > 0.0:
            x = self.forward_features_mask(x, mask_t_prob=mask_t_prob, mask_f_prob=mask_f_prob)
        else:
            x = self.forward_features(x)
        x = self.head(x)
        return x


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


# ── Probe helpers ─────────────────────────────────────────────────────────────

def _mae_encode_layers(self, x):
    """Run AudioMAE encoder and collect per-block outputs. Returns list of 12 [B, L, D]."""
    B = x.shape[0]
    x = self.patch_embed(x)
    x = x + self.pos_embed[:, 1:, :]
    cls_token = self.cls_token + self.pos_embed[:, :1, :]
    cls_tokens = cls_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    x = self.pos_drop(x)
    layer_results = []
    for blk in self.blocks:
        x = blk(x)
        layer_results.append(x)
    return layer_results


class MAELayerProbe(VisionTransformer):
    """Layer-wise linear probe for AudioMAE."""
    def __init__(self, probe_layer=11, pooling='avg', **kwargs):
        super().__init__(**kwargs)
        self.probe_layer = probe_layer
        self.pooling = pooling
        self.head.weight.data.normal_(mean=0.0, std=2.5e-5)
        self.head.bias.data.zero_()

    def _freeze_encoder(self):
        for param in self.parameters():
            param.requires_grad = False
        self.head.weight.requires_grad = True
        self.head.bias.requires_grad = True
        self.fc_norm.weight.requires_grad = True
        self.fc_norm.bias.requires_grad = True

    def adjust_linear_prob_train(self):
        self.eval()
        self.fc_norm.train()
        self.head.train()

    def forward(self, x, mask_t_prob=0.0, mask_f_prob=0.0):
        layer_results = _mae_encode_layers(self, x)
        out = layer_results[self.probe_layer]
        if self.pooling == 'avg':
            f = out[:, 1:, :].mean(dim=1)  # skip CLS
        else:
            f = out[:, 0, :]
        f = self.fc_norm(f)
        return self.head(f)


class MAEWeightedSumProbe(VisionTransformer):
    """SUPERB-style weighted-sum linear probe for AudioMAE."""
    def __init__(self, pooling='avg', **kwargs):
        depth = kwargs.get('depth', 12)
        super().__init__(**kwargs)
        self.pooling = pooling
        self.layer_weights = nn.Parameter(torch.zeros(depth))
        self.head.weight.data.normal_(mean=0.0, std=2.5e-5)
        self.head.bias.data.zero_()

    def _freeze_encoder(self):
        for param in self.parameters():
            param.requires_grad = False
        self.layer_weights.requires_grad = True
        self.head.weight.requires_grad = True
        self.head.bias.requires_grad = True
        self.fc_norm.weight.requires_grad = True
        self.fc_norm.bias.requires_grad = True

    def adjust_linear_prob_train(self):
        self.eval()
        self.fc_norm.train()
        self.head.train()

    def forward(self, x, mask_t_prob=0.0, mask_f_prob=0.0):
        layer_results = _mae_encode_layers(self, x)
        stacked = torch.stack(layer_results, dim=0)  # (depth, B, L, D)
        weights = torch.nn.functional.softmax(self.layer_weights, dim=0)
        weighted = (stacked * weights[:, None, None, None]).sum(dim=0)
        if self.pooling == 'avg':
            f = weighted[:, 1:, :].mean(dim=1)
        else:
            f = weighted[:, 0, :]
        f = self.fc_norm(f)
        return self.head(f)


class MAEAttentiveProbe(VisionTransformer):
    """Attentive probing for AudioMAE."""
    def __init__(self, pooling='avg', num_attn_heads=1, **kwargs):
        embed_dim = kwargs.get('embed_dim', 768)
        depth = kwargs.get('depth', 12)
        super().__init__(**kwargs)
        self.pooling = pooling
        self.num_attn_heads = num_attn_heads
        self.attn_scale = (embed_dim // num_attn_heads) ** -0.5
        self.attn_query = nn.Parameter(torch.randn(1, num_attn_heads, 1, embed_dim // num_attn_heads) * 0.02)
        self.attn_key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.probe_layer_norms = nn.ModuleList([nn.LayerNorm(embed_dim, eps=1.e-6) for _ in range(depth)])
        self.head.weight.data.normal_(mean=0.0, std=2.5e-5)
        self.head.bias.data.zero_()

    def _freeze_encoder(self):
        for param in self.parameters():
            param.requires_grad = False
        for p in ('attn_query', 'attn_key', 'attn_value', 'probe_layer_norms', 'fc_norm', 'head'):
            for name, param in self.named_parameters():
                if name.startswith(p):
                    param.requires_grad = True

    def adjust_linear_prob_train(self):
        self.eval()
        self.fc_norm.train()
        self.head.train()
        for ln in self.probe_layer_norms:
            ln.train()
        self.attn_key.train()
        self.attn_value.train()

    def forward(self, x, mask_t_prob=0.0, mask_f_prob=0.0):
        layer_results = _mae_encode_layers(self, x)
        B = layer_results[0].shape[0]
        pooled = []
        for i, lr in enumerate(layer_results):
            lr = self.probe_layer_norms[i](lr)
            if self.pooling == 'avg':
                pooled.append(lr[:, 1:, :].mean(dim=1))
            else:
                pooled.append(lr[:, 0, :])
        stacked = torch.stack(pooled, dim=1)
        H = self.num_attn_heads
        D_h = stacked.shape[-1] // H
        K = self.attn_key(stacked).view(B, len(layer_results), H, D_h).transpose(1, 2)
        V = self.attn_value(stacked).view(B, len(layer_results), H, D_h).transpose(1, 2)
        Q = self.attn_query.expand(B, -1, -1, -1)
        attn = (Q @ K.transpose(-2, -1)) * self.attn_scale
        attn = torch.nn.functional.softmax(attn, dim=-1)
        out = (attn @ V).squeeze(2).reshape(B, -1)
        f = self.fc_norm(out)
        return self.head(f)


def _load_mae_probe(probe_cls, ckpt_path=None, **kwargs):
    model = probe_cls(**kwargs)
    if ckpt_path is not None:
        current_model_dict = model.state_dict()
        loaded_state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if ckpt_path.endswith('.pth'):
            loaded_state_dict = loaded_state_dict['model']
        filtered_state_dict = {}
        for k, v in loaded_state_dict.items():
            if k in current_model_dict and v.size() == current_model_dict[k].size():
                filtered_state_dict[k] = v
        msg = model.load_state_dict(filtered_state_dict, strict=False)
        print(f"MAE probe loaded from {ckpt_path}: {msg}", flush=True)
    return model


def mae_layer_probe(ckpt_path=None, **kwargs):
    return _load_mae_probe(MAELayerProbe, ckpt_path, **kwargs)

def mae_weighted_sum_probe(ckpt_path=None, **kwargs):
    return _load_mae_probe(MAEWeightedSumProbe, ckpt_path, **kwargs)

def mae_attentive_probe(ckpt_path=None, **kwargs):
    return _load_mae_probe(MAEAttentiveProbe, ckpt_path, **kwargs)


def mae_vit(ckpt_path, **kwargs):
    model = VisionTransformer(**kwargs)
    if ckpt_path is not None:
        current_model_dict = model.state_dict()
        loaded_state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if ckpt_path.endswith('.pth'):
            loaded_state_dict = loaded_state_dict['model']
        filtered_state_dict = {}
        for k, v in loaded_state_dict.items():
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
    return model


