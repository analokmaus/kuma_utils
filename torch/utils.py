import torch
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    if len(xm.get_xla_supported_devices()) > 0:
        XLA = True
    else:
        XLA = False
except ModuleNotFoundError:
    XLA = False


def freeze_module(module):
    for i, param in enumerate(module.parameters()):
        param.requires_grad = False


def get_device(arg):
    if isinstance(arg, torch.device) or \
        (XLA and isinstance(arg, xm.xla_device)):
        pass
    elif arg is None or isinstance(arg, (list, tuple)):
        if XLA:
            device = xm.xla_device()
        else:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(arg, str):
        device = torch.device(arg)

    if isinstance(arg, (list, tuple)):
        if isinstance(arg[0], int):
            device_ids = list(arg)
        elif isinstance(arg[0], str):
            if arg[0].isnumeric():
                device_ids = [ int(a) for a in arg ]
            elif ':' in arg[0]:
                device_ids = [ int(a.split(':')[-1]) for a in arg ]
            else:
                raise ValueError(f'Invalid device: {arg}')
        else:
            raise ValueError(f'Invalid device: {arg}')
    else:
        if device.type == 'cuda':
            assert torch.cuda.is_available()
            if device.index is None:
                device_count = torch.cuda.device_count()
                if device_count > 1:
                    device_ids = list(range(device_count))
                else:
                    device_ids = [0]
            else:
                device_ids = [device.index]
        else:
            device_ids = [device.index]
    
    return device, device_ids
