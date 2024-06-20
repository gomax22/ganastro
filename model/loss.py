# Adapted from https://github.com/victoresque/pytorch-template/blob/master/model/loss.py

import torch.nn.functional as F
import torch

def nll_loss(output, target):
    return F.nll_loss(output, target)

def bce_loss_with_logits(output, target):
    return F.binary_cross_entropy_with_logits(output, target)

def bce_loss(output, target):
    return F.binary_cross_entropy(output, target)

def wasserstein_loss(output, target):
    return -torch.mean(output * target)

def context_loss(real_imgs, fake_imgs, mask):
    unmasked_pixels = torch.sum(mask)
    return torch.sum(torch.abs(real_imgs - fake_imgs) * mask) / unmasked_pixels # minimize l1 loss on unmasked pixels -> return l1 loss
    
    # return F.l1_loss(fake_imgs * mask, real_imgs * mask, reduction='mean') # l1 norm
    
    # return F.mse_loss(fake_imgs, real_imgs, reduction='sum') # l2 norm
    # return torch.sum(torch.abs(real_imgs - fake_imgs) * mask) # l1 norm
    # return torch.sqrt(torch.sum(((real_imgs - fake_imgs) ** 2) * mask)) # l2 norm

def perceptual_loss(preds):
    return torch.sum(-torch.log(F.sigmoid(preds)))

def inpaint_loss(masked_imgs, generated_imgs, masks, fake_preds, real_labels, lamb=0.001):
    c_loss = context_loss(masked_imgs, generated_imgs, masks)
    # prior_loss = perceptual_loss(fake_preds)
    prior_loss = bce_loss_with_logits(fake_preds, real_labels)
    return [c_loss + lamb * prior_loss, c_loss, prior_loss, lamb * prior_loss]
