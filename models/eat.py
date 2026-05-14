from transformers import AutoModel
import torch.nn as nn
import torch

class EATModelWrapper(nn.Module):
    def __init__(self, model_id=None, img_size=(1024, 128), embed_dim=768, num_classes=527, ckpt_path=None, **kwargs):
        super().__init__()
        self.img_size = img_size
        if ckpt_path is not None:
            # Load from local checkpoint (fairseq format)
            self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True).eval()
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            state_dict = ckpt['model'] if 'model' in ckpt else ckpt
            # Remap fairseq checkpoint keys to HuggingFace model keys
            remapped = {}
            for k, v in state_dict.items():
                if k == '_ema':
                    continue
                elif k.startswith('modality_encoders.IMAGE.'):
                    suffix = k.replace('modality_encoders.IMAGE.', '')
                    # context_encoder.norm -> pre_norm
                    if suffix.startswith('context_encoder.norm'):
                        suffix = suffix.replace('context_encoder.norm', 'pre_norm')
                    # Skip decoder weights (not needed for fine-tuning)
                    if suffix.startswith('decoder.'):
                        continue
                    remapped['model.' + suffix] = v
                elif k.startswith('blocks.'):
                    remapped['model.' + k] = v
                else:
                    remapped[k] = v
            current = self.model.state_dict()
            filtered = {}
            for k, v in remapped.items():
                if k in current and v.size() == current[k].size():
                    filtered[k] = v
                elif k in current:
                    print(f"EAT size mismatch: {k} ckpt={v.size()} model={current[k].size()}")
            msg = self.model.load_state_dict(filtered, strict=False)
            print(f"EAT loaded from {ckpt_path}: {msg}")
        else:
            self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True).eval()
        self.blocks = self.model.model.blocks
        # Official EAT: fc_norm + linear head on CLS token
        self.fc_norm = nn.LayerNorm(embed_dim, eps=1.e-6)
        nn.init.constant_(self.fc_norm.bias, 0)
        nn.init.constant_(self.fc_norm.weight, 1.0)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.constant_(self.head.bias, 0)
            
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

    def encode(self, x, mask_t_prob=0.0, mask_f_prob=0.0):
        B = x.shape[0]
        x = self.model.model.local_encoder(x)
        if self.model.model.fixed_positional_encoder is not None:
            x = x + self.model.model.fixed_positional_encoder(x, None)[:, :x.size(1), :]
        x = torch.cat((self.model.model.extra_tokens.expand(B, -1, -1), x), dim=1)
        x = self.model.model.pre_norm(x)
        x = self.model.model.pos_drop(x)

        mask, ids_restore, ids_keep = None, None, None
        B, L, D = x.shape
        if self.training:
            cls, x = x[:, :1, :], x[:, 1:, :]
            x, _, _ = self.random_masking_2d(x, mask_t_prob, mask_f_prob)
            x = torch.cat((cls, x), dim=1)

        for blk in self.model.model.blocks:
            x, _ = blk(x)
        return x
    
    def no_weight_decay(self):
        """Set of parameters that should not use weight decay."""
        return {'pos_embed', 'cls_token', 'dist_token'}
    
    def adjust_linear_prob_train(self):
        # Set model to training mode except for fc_norm and head
        self.model.eval()
        self.fc_norm.train()
        self.head.train()

    def _linear_prob_freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.head.weight.requires_grad = True
        self.head.bias.requires_grad = True
        self.fc_norm.weight.requires_grad = True
        self.fc_norm.bias.requires_grad = True

    def forward(self, x, mask_t_prob=0.0, mask_f_prob=0.0):
        features = self.encode(x, mask_t_prob, mask_f_prob)
        # CLS token pooling (official EAT: prediction_mode=CLS_TOKEN)
        features = features[:, 0]
        features = self.fc_norm(features)
        logits = self.head(features)
        return logits

def eat_classifier(**kwargs):
    return EATModelWrapper(**kwargs)


# ── Probe helpers ─────────────────────────────────────────────────────────────

def _eat_encode_layers(self, x):
    """Run encoder and collect per-block outputs. Returns list of 12 [B, L, D] tensors."""
    B = x.shape[0]
    x = self.model.model.local_encoder(x)
    if self.model.model.fixed_positional_encoder is not None:
        x = x + self.model.model.fixed_positional_encoder(x, None)[:, :x.size(1), :]
    x = torch.cat((self.model.model.extra_tokens.expand(B, -1, -1), x), dim=1)
    x = self.model.model.pre_norm(x)
    x = self.model.model.pos_drop(x)
    layer_results = []
    for blk in self.model.model.blocks:
        x, _ = blk(x)
        layer_results.append(x)
    return layer_results


class EATLayerProbe(EATModelWrapper):
    """Layer-wise linear probe for EAT / SSLAM."""
    def __init__(self, probe_layer=11, pooling='avg', **kwargs):
        kwargs.pop('mask_t_prob', None)
        kwargs.pop('mask_f_prob', None)
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

    def forward(self, x, mask_t_prob=0.0, mask_f_prob=0.0):
        layer_results = _eat_encode_layers(self, x)
        out = layer_results[self.probe_layer]
        if self.pooling == 'avg':
            f = out.mean(dim=1)
        else:
            f = out[:, 0, :]
        f = self.fc_norm(f)
        return self.head(f)


class EATWeightedSumProbe(EATModelWrapper):
    """SUPERB-style weighted-sum linear probe for EAT / SSLAM."""
    def __init__(self, depth=12, pooling='avg', **kwargs):
        kwargs.pop('mask_t_prob', None)
        kwargs.pop('mask_f_prob', None)
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
        self.model.eval()
        self.fc_norm.train()
        self.head.train()

    def forward(self, x, mask_t_prob=0.0, mask_f_prob=0.0):
        layer_results = _eat_encode_layers(self, x)
        stacked = torch.stack(layer_results, dim=0)  # (depth, B, L, D)
        weights = torch.nn.functional.softmax(self.layer_weights, dim=0)
        weighted = (stacked * weights[:, None, None, None]).sum(dim=0)  # (B, L, D)
        if self.pooling == 'avg':
            f = weighted.mean(dim=1)
        else:
            f = weighted[:, 0, :]
        f = self.fc_norm(f)
        return self.head(f)


class EATAttentiveProbe(EATModelWrapper):
    """Attentive probing for EAT / SSLAM."""
    def __init__(self, depth=12, pooling='avg', num_attn_heads=1, **kwargs):
        kwargs.pop('mask_t_prob', None)
        kwargs.pop('mask_f_prob', None)
        embed_dim = kwargs.get('embed_dim', 768)
        super().__init__(**kwargs)
        self.pooling = pooling
        self.num_attn_heads = num_attn_heads
        self.attn_scale = (embed_dim // num_attn_heads) ** -0.5
        self.attn_query = nn.Parameter(torch.randn(1, num_attn_heads, 1, embed_dim // num_attn_heads) * 0.02)
        self.attn_key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(embed_dim, eps=1.e-6) for _ in range(depth)])
        self.head.weight.data.normal_(mean=0.0, std=2.5e-5)
        self.head.bias.data.zero_()

    def _freeze_encoder(self):
        for param in self.parameters():
            param.requires_grad = False
        for p in ('attn_query', 'attn_key', 'attn_value', 'layer_norms', 'fc_norm', 'head'):
            for name, param in self.named_parameters():
                if name.startswith(p):
                    param.requires_grad = True

    def adjust_linear_prob_train(self):
        self.model.eval()
        self.fc_norm.train()
        self.head.train()
        for ln in self.layer_norms:
            ln.train()
        self.attn_key.train()
        self.attn_value.train()

    def forward(self, x, mask_t_prob=0.0, mask_f_prob=0.0):
        layer_results = _eat_encode_layers(self, x)
        B = layer_results[0].shape[0]
        pooled = []
        for i, lr in enumerate(layer_results):
            lr = self.layer_norms[i](lr)
            if self.pooling == 'avg':
                pooled.append(lr.mean(dim=1))
            else:
                pooled.append(lr[:, 0, :])
        stacked = torch.stack(pooled, dim=1)  # (B, num_layers, D)
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


def eat_layer_probe(**kwargs):
    return EATLayerProbe(**kwargs)

def eat_weighted_sum_probe(**kwargs):
    return EATWeightedSumProbe(**kwargs)

def eat_attentive_probe(**kwargs):
    return EATAttentiveProbe(**kwargs)
