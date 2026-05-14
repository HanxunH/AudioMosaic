import mlconfig
import torch
from .ntxent import NTXentLoss
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

mlconfig.register(LabelSmoothingCrossEntropy)
mlconfig.register(SoftTargetCrossEntropy)
mlconfig.register(torch.nn.BCEWithLogitsLoss)
mlconfig.register(torch.nn.CrossEntropyLoss)
mlconfig.register(NTXentLoss)