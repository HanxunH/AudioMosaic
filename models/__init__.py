import mlconfig
import torch
from .mae_vit import mae_vit, mae_layer_probe, mae_weighted_sum_probe, mae_attentive_probe
from .audiomosaic import audiomosaic_pretrain, audiomosaic_classifier, audiomosaic_aasist, audiomosaic_weighted_sum_probe, audiomosaic_layer_probe, audiomosaic_attentive_probe
from .eat import eat_classifier, eat_layer_probe, eat_weighted_sum_probe, eat_attentive_probe
from .beats import beats_classifier, beats_layer_probe, beats_weighted_sum_probe, beats_attentive_probe
from .aist import aist_classifier

mlconfig.register(mae_vit)
mlconfig.register(mae_layer_probe)
mlconfig.register(mae_weighted_sum_probe)
mlconfig.register(mae_attentive_probe)
mlconfig.register(audiomosaic_pretrain)
mlconfig.register(audiomosaic_classifier)
mlconfig.register(audiomosaic_aasist)
mlconfig.register(audiomosaic_weighted_sum_probe)
mlconfig.register(audiomosaic_layer_probe)
mlconfig.register(audiomosaic_attentive_probe)
mlconfig.register(eat_classifier)
mlconfig.register(eat_layer_probe)
mlconfig.register(eat_weighted_sum_probe)
mlconfig.register(eat_attentive_probe)
mlconfig.register(beats_classifier)
mlconfig.register(beats_layer_probe)
mlconfig.register(beats_weighted_sum_probe)
mlconfig.register(beats_attentive_probe)
mlconfig.register(aist_classifier)
mlconfig.register(torch.optim.SGD)
mlconfig.register(torch.optim.Adam)
mlconfig.register(torch.optim.AdamW)
mlconfig.register(torch.optim.LBFGS)
mlconfig.register(torch.optim.lr_scheduler.MultiStepLR)
mlconfig.register(torch.optim.lr_scheduler.CosineAnnealingLR)
mlconfig.register(torch.optim.lr_scheduler.StepLR)
mlconfig.register(torch.optim.lr_scheduler.ExponentialLR)