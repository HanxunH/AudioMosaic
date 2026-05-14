import math
from functools import partial

import numpy
import torch
import torch.nn as nn

from torch.nn.init import trunc_normal_


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=[1024, 128], patch_size=[16, 16], in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.patch_hw = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, audio_size=[1024, 128], patch_size=[16, 16], in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.audio_size = audio_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            img_size=audio_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = (w/16)*(h/16)
        N = self.pos_embed.shape[1] - 1
        if npatch == N:
            return self.pos_embed
        
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        
        sz1 = w//self.patch_size[0]
        sz2 = h//self.patch_size[0]
        
        prev_sz1 = self.audio_size[0]//self.patch_size[0]
        prev_sz2 = self.audio_size[1]//self.patch_size[1]
        patch_pos_embed = torch.nn.functional.interpolate(
            patch_pos_embed.transpose(1, 2).reshape(1, self.embed_dim, prev_sz1, prev_sz2), size=(sz1, sz2), mode='bicubic', align_corners=False)
        

        patch_pos_embed = patch_pos_embed.reshape(1, self.embed_dim, sz1*sz2).transpose(1, 2)

        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)
        #x = x + self.pos_embed
        return self.pos_drop(x)

    def forward(self, x, classify=False):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if classify==True:
            return self.head(x[:, 0])
        return x

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

class AISTClassifier(VisionTransformer):
    def __init__(self, patch_size=[16, 16], audio_size=[1024, 128], in_chans=3, num_classes=400, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__(audio_size=audio_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                         embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate, norm_layer=norm_layer, **kwargs)
        
        self.head = nn.Linear(embed_dim, num_classes)
        self.head.weight.data.normal_(mean=0.0, std=2.5e-5)
        self.head.bias.data.zero_()
        self.fc_norm = nn.LayerNorm(embed_dim, eps=1.e-6)
        nn.init.constant_(self.fc_norm.bias, 0)
        nn.init.constant_(self.fc_norm.weight, 1.0)

    def no_weight_decay(self):
        """Set of parameters that should not use weight decay."""
        return {'pos_embed', 'cls_token', 'dist_token'}
    
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
    
    def prepare_tokens(self, x, mask_t_prob=0.0, mask_f_prob=0.0):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)
        #x = x + self.pos_embed
        x = self.pos_drop(x)
        if self.training:
            cls, x = x[:, :1, :], x[:, 1:, :]
            x, _, _ = self.random_masking_2d(x, mask_t_prob, mask_f_prob)
            x = torch.cat((cls, x), dim=1)
        return x
    
    def forward(self, x, mask_t_prob=0.0, mask_f_prob=0.0):
        x = self.prepare_tokens(x, mask_t_prob, mask_f_prob)
        for blk in self.blocks:
            x = blk(x)
        x = self.fc_norm(x)
        x = self.head(x[:, 0])
        return x


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small(patch_size=[16, 16], audio_size=[1024, 128], stride=[16, 16], **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, audio_size=audio_size, stride=stride, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base(patch_size=[16, 16], audio_size=[1024, 128], stride=[16, 16], **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, audio_size=audio_size, stride=stride, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
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


def aist_classifier(ckpt_path, **kwargs):
    model = AISTClassifier(
        patch_size=[16, 16], audio_size=[1024, 128], embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    current_model_dict = model.state_dict()
    loaded_state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)["student"]
    filtered_state_dict = {}
    for k, v in loaded_state_dict.items():
        if "module.backbone." in k:
            k = k.replace("module.backbone.", "")
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
    print('load aist classifier from %s'%ckpt_path)
    print(msg)
    return model

class CLSHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


class RECHead(nn.Module):
    def __init__(self, in_dim, audio_size, in_chans=3, patch_size=16):
        super().__init__()
        
        self.audio_size = audio_size
        self.patch_size = patch_size

        layers = [nn.Linear(in_dim, in_dim)]
        layers.append(nn.GELU())
        layers.append(nn.Linear(in_dim, in_dim))
        layers.append(nn.GELU())
        layers.append(nn.Linear(in_dim, in_dim))
        layers.append(nn.GELU())

        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

        self.convTrans = nn.ConvTranspose2d(in_dim, in_chans, kernel_size=(patch_size, patch_size),
                                                stride=(patch_size, patch_size))
        


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)

        x_rec = x.transpose(1, 2)
        out_sz = (self.audio_size[0]//self.patch_size , self.audio_size[1]//self.patch_size ) #tuple( (  int(math.sqrt(x_rec.size()[2]))  ,   int(math.sqrt(x_rec.size()[2])) ) )
        x_rec = self.convTrans(x_rec.unflatten(2, out_sz))


        return x_rec