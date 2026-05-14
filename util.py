import logging
import os
import numpy as np
import torch
import json
import math
import torch.nn as nn
import json
import os.path as osp
from typing import Union
from torch import inf
from decimal import Decimal
from sklearn import metrics
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("../templates", f"{template_name}.json")
        if not osp.exists(file_name):
            file_name = osp.join("./templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
    

def round_num(n, decimal=2):
    n = Decimal(n)
    return round(n.normalize(), decimal)


def numerize(n, decimal=2):
    # https://github.com/davidsa03/numerize/blob/master/numerize/numerize.py
    #60 sufixes
    sufixes = [ "", "K", "M", "B", "T", "Qa", "Qu", "S", "Oc", "No", 
                "D", "Ud", "Dd", "Td", "Qt", "Qi", "Se", "Od", "Nd","V", 
                "Uv", "Dv", "Tv", "Qv", "Qx", "Sx", "Ox", "Nx", "Tn", "Qa",
                "Qu", "S", "Oc", "No", "D", "Ud", "Dd", "Td", "Qt", "Qi",
                "Se", "Od", "Nd", "V", "Uv", "Dv", "Tv", "Qv", "Qx", "Sx",
                "Ox", "Nx", "Tn", "x", "xx", "xxx", "X", "XX", "XXX", "END"] 
    
    sci_expr = [1e0, 1e3, 1e6, 1e9, 1e12, 1e15, 1e18, 1e21, 1e24, 1e27, 
                1e30, 1e33, 1e36, 1e39, 1e42, 1e45, 1e48, 1e51, 1e54, 1e57, 
                1e60, 1e63, 1e66, 1e69, 1e72, 1e75, 1e78, 1e81, 1e84, 1e87, 
                1e90, 1e93, 1e96, 1e99, 1e102, 1e105, 1e108, 1e111, 1e114, 1e117, 
                1e120, 1e123, 1e126, 1e129, 1e132, 1e135, 1e138, 1e141, 1e144, 1e147, 
                1e150, 1e153, 1e156, 1e159, 1e162, 1e165, 1e168, 1e171, 1e174, 1e177]
    minus_buff = n
    n = abs(n)
    if n == 0:
        return str(n)
    for x in range(len(sci_expr)):
        try:
            if n >= sci_expr[x] and n < sci_expr[x+1]:
                sufix = sufixes[x]
                if n >= 1e3:
                    num = str(round_num(n/sci_expr[x], decimal))
                else:
                    num = str(n)
                return num + sufix if minus_buff > 0 else "-" + num + sufix
        except IndexError:
            print("You've reached the end")
    

def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'predictor' in n:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    Supports: AudioMosaic (blocks.X), EAT (model.model.blocks.X), BEATs (encoder.layers.X)
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed') or name.startswith('model.model.local_encoder') or name.startswith('model.model.fixed_positional_encoder'):
        return 0
    elif name.startswith('model.model.extra_tokens') or name.startswith('model.model.pre_norm'):
        return 0
    elif name.startswith('blocks.'):
        return int(name.split('.')[1]) + 1
    elif name.startswith('model.model.blocks.'):
        # EAT: model.model.blocks.X.* -> layer X+1
        return int(name.split('.')[3]) + 1
    elif name.startswith('encoder.layers.'):
        # BEATs: encoder.layers.X.* -> layer X+1
        return int(name.split('.')[2]) + 1
    else:
        return num_layers


def adjust_learning_rate_with_params(optimizer, epoch, min_lr, lr, warmup, epochs, lr_schedule=None):
    if epoch < warmup:
        lr = lr * epoch / warmup
    elif lr_schedule:
        if lr_schedule == 'milestone':
            milestones = [int(s*epochs)for s in [0.65, 0.85, 0.95]]
            if epoch < milestones[0]:
                lr = lr
            elif epoch >= milestones[0] and epoch < milestones[1]:
                lr = lr * 0.1
            else:
                lr = lr * 0.01  
        elif lr_schedule == 'linear':
            lr = lr * (1 - (epoch - warmup) / (epochs - warmup))
        elif lr_schedule == 'cosine':
            lr = min_lr + (lr - min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - warmup) / (epochs - warmup)))
        elif lr_schedule == 'polynomial':
            lr_range = lr - min_lr
            pct_remaining = 1 - (epoch - warmup) / (epochs - warmup)
            lr = lr_range * pct_remaining + min_lr
    else:
        lr = min_lr + (lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup) / (epochs - warmup)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        elif 'fix_lr' in param_group and param_group['fix_lr']:
            param_group["lr"] = lr
        else:
            param_group["lr"] = lr
    return lr


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def log_display(global_step, time_elapse, epoch=None, **kwargs):
    display = 'global_step=' + str(numerize(global_step, 1))
    for key, value in kwargs.items():
        if type(value) == str:
            display = ' ' + key + '=' + value
        else:
            if 'acc' in key: 
                display += ' ' + str(key) + '=%.4f' % value
            elif 'loss' in key:
                display += ' ' + str(key) + '=%.4f' % value
            elif 'lr' in key:
                display += ' ' + str(key) + '=%.2e' % value
            else:
                display += ' ' + str(key) + '=%.2f' % value
    display += ' time=%.1fit/s' % (1. / time_elapse)
    return display

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)

    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(1/batch_size))
    return res

def count_parameters_in_MB(model):
    return sum(np.prod(v.size()) for name, v in model.named_parameters())/1e6

def build_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = max(self.max, val)

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model
    
def get_gpu_model():
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name().lower()
    if "a100" in name:
        return "A100"
    elif "h100" in name:
        return "H100"
    else:
        # fallback on compute capability
        major, _ = torch.cuda.get_device_capability()
        if major == 8:
            return "A100"
        elif major == 9:
            return "H100"
    return None

def calculate_stats(output, target):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)

    Returns:
      stats: list of statistic of each class.
    """

    classes_num = target.shape[-1]
    stats = []

    # Accuracy, only used for single-label classification such as esc-50, not for multiple label one such as AudioSet
    acc = metrics.accuracy_score(np.argmax(target, 1), np.argmax(output, 1))

    # Class-wise statistics
    for k in range(classes_num):

        # Average precision
        avg_precision = metrics.average_precision_score(
            target[:, k], output[:, k], average=None)

        # AUC
        auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)

        # Precisions, recalls
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(
            target[:, k], output[:, k])

        # FPR, TPR
        (fpr, tpr, thresholds) = metrics.roc_curve(target[:, k], output[:, k])

        save_every_steps = 1000     # Sample statistics to reduce size
        dict = {'precisions': precisions[0::save_every_steps],
                'recalls': recalls[0::save_every_steps],
                'AP': avg_precision,
                'fpr': fpr[0::save_every_steps],
                'fnr': 1. - tpr[0::save_every_steps],
                'auc': auc,
                # note acc is not class-wise, this is just to keep consistent with other metrics
                'acc': acc
                }
        stats.append(dict)

    return stats