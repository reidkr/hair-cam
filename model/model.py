import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18 as _resnet18


def resnet18(num_classes, use_pretrained=False, **kwargs):
    '''
    Resnet18 model
    :param pretrained: Bool, whether to load pretrained weights into model
    :param num_classes: Int, number of class labels
    '''
    # call model, load pretrained weights
    model = _resnet18(use_pretrained, **kwargs)
    # reinitialize final fc layer
    in_ftrs = model.fc.in_features
    model.fc = nn.Linear(in_features=in_ftrs, out_features=num_classes)
    return model
