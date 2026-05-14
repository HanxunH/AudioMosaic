import misc
import torch
from torch import nn
from lid import lid_mle, lid_mom_est
import torch.nn.functional as F
import util
import torch.distributed as dist
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class NTXentLoss(nn.Module):
    def __init__(self, 
                 temperature: float = 0.5, 
                 gather_distributed: bool = False, 
                 learnable_tau=False, 
                 learnable_bias=False, 
                 online_classifier=False,
                 tf_masking=False,
                 masking_mode="none"
                 ):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.gather_distributed = gather_distributed
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.eps = 1e-8
        self.learnable_tau = learnable_tau
        self.learnable_bias = learnable_bias
        self.online_classifier = online_classifier
        self.masking_mode = masking_mode
        if tf_masking:
            self.masking_mode = "tf" # Legacy support

    def track_lid(self, f_0, f_1, k=32):
        # Track LID
        with torch.no_grad():
            f = torch.cat([f_0.float(), f_1.float()], dim=0).detach()
            if self.gather_distributed:
                full_rank_f = torch.cat(misc.gather(f), dim=0)
            else:
                full_rank_f = f
            lids = lid_mom_est(data=f.detach(), reference=full_rank_f.detach(), k=k)
        return lids
    
    def forward(self, model, x, y=None):
        x0, x1 = x  # two augmented versions
        x0 = x0.to(device, non_blocking=True)
        x1 = x1.to(device, non_blocking=True)
        if y is not None:
            y = y.to(device, non_blocking=True)

        if self.masking_mode == "tf":
            results_0 = model({"x": x0, "branch": "time"})
            results_1 = model({"x": x1, "branch": "freq"})
        elif self.masking_mode == "tt":
            results_0 = model({"x": x0, "branch": "time"})
            results_1 = model({"x": x1, "branch": "time"})
        elif self.masking_mode == "ff":
            results_0 = model({"x": x0, "branch": "freq"})
            results_1 = model({"x": x1, "branch": "freq"})
        elif self.masking_mode == "unsturctured":
            results_0 = model({"x": x0, "branch": "unsturctured"})
            results_1 = model({"x": x1, "branch": "unsturctured"})
        else:
            results_0 = model(x0)
            results_1 = model(x1)

        z_0 = results_0['z']
        z_1 = results_1['z']        

        batch_size = z_0.shape[0]
        z_0 = F.normalize(z_0, dim=1)
        z_1 = F.normalize(z_1, dim=1)
        
        if self.learnable_tau:
            logits_scale = results_0['logit_scale'] + results_1['logit_scale'] 
            logits_scale = logits_scale / 2.0
            temperature = 1.0 / logits_scale
        else:   
            temperature = self.temperature
        # user other samples from batch as negatives
        # and create diagonal mask that only selects similarities between
        # views of the same image
        if self.gather_distributed and misc.world_size() > 1:
            # gather hidden representations from other processes
            out0_large = torch.cat(misc.gather(z_0), 0)
            out1_large = torch.cat(misc.gather(z_1), 0)
            diag_mask = misc.eye_rank(batch_size, device=z_0.device)
        else:
            # single process
            out0_large = z_0
            out1_large = z_1
            diag_mask = torch.eye(batch_size, device=z_0.device, dtype=torch.bool)
        
        # calculate similiarities
        # here n = batch_size and m = batch_size * world_size
        # the resulting vectors have shape (n, m)
        logits_00 = torch.einsum('nc,mc->nm', z_0, out0_large) / temperature
        logits_01 = torch.einsum('nc,mc->nm', z_0, out1_large) / temperature
        logits_10 = torch.einsum('nc,mc->nm', z_1, out0_large) / temperature
        logits_11 = torch.einsum('nc,mc->nm', z_1, out1_large) / temperature

        # remove simliarities between same views of the same image
        logits_00 = logits_00[~diag_mask].view(batch_size, -1)
        logits_11 = logits_11[~diag_mask].view(batch_size, -1)

        # concatenate logits
        # the logits tensor in the end has shape (2*n, 2*m-1)
        logits_0100 = torch.cat([logits_01, logits_00], dim=1)
        logits_1011 = torch.cat([logits_10, logits_11], dim=1)
        logits = torch.cat([logits_0100, logits_1011], dim=0)
        if self.learnable_bias:
            logits_bias_0 = results_0['logit_bias']
            logits_bias_1 = results_1['logit_bias']
            logits_bias = (logits_bias_0 + logits_bias_1) / 2.0
            logits = logits + logits_bias
        # create labels
        labels = torch.arange(batch_size, device=z_0.device, dtype=torch.long)
        labels = labels + misc.rank() * batch_size
        labels = labels.repeat(2)
        
        clr_loss = self.cross_entropy(logits, labels)
        loss = clr_loss.mean(dim=0)

        if self.online_classifier:
            online_logits_0 = results_0['online_logits']
            online_logits_1 = results_1['online_logits']
            online_logits = torch.cat([online_logits_0, online_logits_1], dim=0)
            online_labels = torch.cat([y, y], dim=0)
            online_loss = self.bce_loss(online_logits, online_labels.float())
            loss = loss + online_loss

        # Track LID
        f_0 = results_0['f']
        f_1 = results_1['f']
        lids = self.track_lid(f_0, f_1)
        acc = util.accuracy(logits, labels, topk=(1,))[0]

        results = {
            "loss": loss,
            "acc": acc.mean().item(),
            "lids_mean": lids.detach().mean().item(),
            "lids_var": lids.detach().var().item(),
            "main_loss": clr_loss.mean().item(),
            "temperature": temperature.item() if torch.is_tensor(temperature) else temperature,
        }
        if self.online_classifier:
            results["online_loss"] = online_loss.item()
        if self.learnable_tau:
            results['logit_scale'] = logits_scale.item()
        if self.learnable_bias:
            results['logit_bias'] = logits_bias.item()
        return results

