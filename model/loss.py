# Adapted from https://github.com/victoresque/pytorch-template/blob/master/model/loss.py

import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

def bce_loss_with_logits(output, target):
    return F.binary_cross_entropy_with_logits(output, target)

def bce_loss(output, target):
    return F.binary_cross_entropy(output, target)