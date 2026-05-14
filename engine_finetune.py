
import torch
import numpy as np
import torch.nn.functional as F
import misc
import time
from collections import defaultdict
from sklearn import metrics
from sklearn.metrics import auc, average_precision_score, roc_curve
import util
import math
import warnings

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def get_grad_norm(params, scale=1):
    """Compute grad norm given a gradient scale."""
    nan_detected = False
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            param_norm = (p.grad.detach().data / scale).norm(2, dtype=torch.float32)
            if torch.isnan(param_norm).any() or torch.isinf(param_norm).any():
                nan_detected = True
                p.grad.data.zero_()
                param_norm = (p.grad.detach().data / scale).norm(2, dtype=torch.float32)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm, nan_detected


def train_step(exp, x, y, model, optimizer, scaler, criterion):
    # Train step
    optimizer.zero_grad()
    results = defaultdict(float)

    with torch.amp.autocast(enabled=scaler is not None, device_type=device.type):
        if hasattr(exp.config, "mask_t_prob"):
            logits = model(x, mask_t_prob=exp.config.mask_t_prob, mask_f_prob=exp.config.mask_f_prob)
        else:
            logits = model(x)
        loss = criterion(logits, y)

    if scaler is not None:
        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()

    if hasattr(exp.config, 'grad_clip'):
        torch.nn.utils.clip_grad_norm_(model.parameters(), exp.config.grad_clip)
    grad_norm, nan_detected = get_grad_norm(model.parameters())

    if nan_detected:
        warnings.warn("NaN detected in grad norm")
    if scaler is not None:
        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)
        # Updates the scale for next iteration.
        scaler.update()
    else:
        optimizer.step()
    if y.ndim == 1:
        acc = util.accuracy(logits.detach(), y)[0]
        results['acc'] = acc.item()
    results['loss'] = loss.item()
    results['grad_norm'] = grad_norm
    torch.cuda.synchronize()
    return results


def train_epoch(exp, global_step, data_loader, model, optimizer, scaler, criterion, logger):
    # Set Meters
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    for x, y in data_loader:
        start_time = time.time()
        warmup_step = len(data_loader) * exp.config.warmup_epochs
        util.adjust_learning_rate_with_params(
            optimizer=optimizer, epoch=global_step, warmup=warmup_step, epochs=exp.config.epochs * len(data_loader), 
            min_lr=exp.config.min_lr, lr=exp.config.lr, lr_schedule=exp.config.lr_schedule
        )
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        stats = train_step(exp, x, y, model, optimizer, scaler, criterion)
        end_time = time.time()
        time_used = end_time - start_time
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        lr = max_lr
        # Logging
        batch_size = x[0].shape[0]
        for k in stats.keys():
            if k == 'batch_size' or k == "n" or k == 'lr':
                continue
            if k in stats:
                if type(stats[k]) == torch.Tensor:
                    v = stats[k].item()
                else:
                    v = stats[k]
                metric_logger.update(**{k: v}, n=batch_size)
        
        if global_step % exp.config.log_frequency == 0:
            metric_logger.synchronize_between_processes()
            payload = {
                "lr": lr,
            }
            online_log_payload = {
                "lr": lr
            }
            for k, v in metric_logger.meters.items():
                if k == 'batch_size' or k == "n" or k == 'lr':
                    continue
                payload[k] = v.avg
                online_log_payload[k] = v.avg
            if misc.get_rank() == 0:
                display = util.log_display(
                    global_step=global_step,
                    time_elapse=time_used,
                    **payload
                )
                logger.info(display)
                exp.online_log(online_log_payload, step=global_step)

        global_step += 1

    metric_logger.synchronize_between_processes()
    train_stats = {
        "global_step": global_step,
        "lr": lr,
        "loss": metric_logger.loss.avg,
        "grad_norm": metric_logger.grad_norm.avg,
    }
    return train_stats
            
           

@torch.no_grad()
def evaluate(model, loader):
    # Evaluate
    model.eval()

    outputs = []
    targets = []
    N = 0
    true_y = []
    for i, data in enumerate(loader):
        x, y = data
        x, y = x.to(device), y.to(device)
        out = model(x)        
        N += out.size(0)
        if y.ndim == 1 or out.shape[1] == 2:
            outputs.append(out.detach().cpu())
        else:
            outputs.append(torch.sigmoid(out).detach().cpu())
        targets.append(y.detach().cpu())

    if out.shape[1] == 2:
        # Deepfake detection with 2 classes
        outputs = torch.cat(outputs, dim=0)
        targets = torch.cat(targets, dim=0)
        # ROC expects a 1D score per sample; use positive-class probability
        pos_scores = torch.softmax(outputs, dim=1)[:, 1]
        fpr, tpr, threshold = roc_curve(targets, pos_scores, pos_label=1)
        roc_auc = auc(fpr, tpr)
        fnr = 1 - tpr
        eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
        EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        stats = {
            'EER': float(EER),
            'eer_threshold': float(eer_threshold),
            'AUC': float(roc_auc),
            "N": N
        }
    elif y.ndim == 1:
        outputs = torch.cat(outputs, dim=0)
        targets = torch.cat(targets, dim=0)
        acc = util.accuracy(outputs, targets)[0]
        stats = {
            'acc': float(acc),
            "N": N
        }
    else:
        outputs = torch.cat(outputs, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()
        stats = util.calculate_stats(outputs, targets)
        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        acc = np.mean([stat['acc'] for stat in stats])
        
        stats = {
            'mAP': float(mAP),
            'mAUC': float(mAUC),
            'acc': float(acc),
            "N": N
        }
    return stats
