# Adapted from https://github.com/victoresque/pytorch-template/blob/master/model/loss.py

import torch.nn.functional as F
import torch

def nll_loss(output, target):
    return F.nll_loss(output, target)

def bce_loss_with_logits(output, target):
    return F.binary_cross_entropy_with_logits(output, target)

def bce_loss(output, target):
    return F.binary_cross_entropy(output, target)

def wasserstein(output, target):
    return -torch.mean(output * target)

def context_loss(corrupted_images, generated_images, masks):
    # return torch.sum(torch.abs(corrupted_images - generated_images) * masks) # l1 norm
    return torch.sqrt(torch.sum((corrupted_images - generated_images) ** 2 * masks)) # l2 norm

def perceptual_loss(preds):
    return torch.sum(-torch.log(F.sigmoid(preds)))

def inpaint_loss(masked_imgs, generated_imgs, masks, preds, lamb=0.001):
    c_loss = context_loss(masked_imgs, generated_imgs, masks)
    prior_loss = perceptual_loss(preds)
    return [c_loss + lamb * prior_loss, c_loss, prior_loss, lamb * prior_loss]
