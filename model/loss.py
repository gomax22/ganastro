# Adapted from https://github.com/victoresque/pytorch-template/blob/master/model/loss.py

import torch.nn.functional as F
import torch

MASKING_LEVELS = {60: (11, 44), 40: (17, 39), 20: (22, 33)}


def nll_loss(output, target):
    return F.nll_loss(output, target)

def bce_loss_with_logits(output, target):
    return F.binary_cross_entropy_with_logits(output[:target.size(0)], target)

def bce_loss(output, target):
    return F.binary_cross_entropy(output, target)

def wasserstein_loss(output, target):
    return -torch.mean(output * target)

def context_loss(real_imgs, fake_imgs, mask):
    return F.mse_loss(fake_imgs[:real_imgs.size(0)] * mask, real_imgs * mask, reduction='sum')

def perceptual_loss(preds):
    return torch.sum(-torch.log(F.sigmoid(preds)))

def inpaint_loss(masked_imgs, generated_imgs, masks, fake_preds, real_labels, lamb=0.001):
    c_loss = context_loss(masked_imgs, generated_imgs, masks)
    prior_loss = bce_loss_with_logits(fake_preds, real_labels)
    return [(1 - lamb) * c_loss + lamb * prior_loss, (1 - lamb) * c_loss, prior_loss, lamb * prior_loss]

def recon_loss(real_imgs, fake_imgs, masking_level=60):
    b, c, h, w = real_imgs.size()
    
    b1, b2 = MASKING_LEVELS[masking_level]
    
    upper_loss = F.l1_loss(fake_imgs[:, :, :b1, :], real_imgs[:, :, :b1, :], reduction='mean')
    middle_loss = F.l1_loss(fake_imgs[:, :, b1:b2, :], real_imgs[:, :, b1:b2, :], reduction='mean')
    lower_loss = F.l1_loss(fake_imgs[:, :, b2:, :], real_imgs[:, :, b2:, :], reduction='mean')
    
    return upper_loss, middle_loss, lower_loss
    
def inpaint_loss_v2(real_imgs, generated_imgs, fake_preds, real_labels, alpha, beta, lamb=0.001, masking_level=60):
    upper_loss, middle_loss, lower_loss = recon_loss(real_imgs, generated_imgs)
    prior_loss = bce_loss_with_logits(fake_preds, real_labels)
    
    return [alpha * (upper_loss + lower_loss) + beta * middle_loss + lamb * prior_loss, # global loss
            upper_loss + lower_loss, # upper loss
            middle_loss, # inside transit
            lamb * prior_loss] # prior loss weigthed

def l1_loss(recons, target):
    return F.l1_loss(recons, target, reduction='mean')


