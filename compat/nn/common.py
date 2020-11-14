import torch


def scan_requires_grad(model):
    frozen = 0
    unfrozen = 0
    for i, param in enumerate(model.parameters()):
        if param.requires_grad:
            unfrozen += 1
        else:
            frozen += 1
    return frozen, unfrozen


def set_requires_grad(model, requires_grad=True, verbose=True):
    for i, param in enumerate(model.parameters()):
        param.requires_grad = requires_grad
    if verbose:
        frozen, unfrozen = scan_requires_grad(model)
        print(f'{frozen}/{frozen+unfrozen} params is frozen.')
