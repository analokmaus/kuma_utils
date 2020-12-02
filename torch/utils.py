import torch


def freeze_module(module):
    for i, param in enumerate(module.parameters()):
        param.requires_grad = False
