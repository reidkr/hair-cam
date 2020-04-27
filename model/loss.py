import torch.nn as nn
import torch.nn.functional as F


def neg_like_loss(output, target):
    '''negative likelihood loss'''
    return F.nll_loss(output, target)


def cross_entropy_loss(output, target, weights=None):
    '''cross entropy loss'''
    if weights is not None:
        # TODO: Add logic for weights
        return nn.CrossEntropyLoss(weight=weights)(output, target)
    else:
        return nn.CrossEntropyLoss()(output, target)
