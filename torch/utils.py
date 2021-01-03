import torch
import random
import subprocess
import time
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    XLA = True
except ModuleNotFoundError:
    XLA = False


def freeze_module(module):
    for i, param in enumerate(module.parameters()):
        param.requires_grad = False


def get_device(arg):
    if isinstance(arg, torch.device) or \
        (XLA and isinstance(arg, xm.xla_device)):
        device = arg
    elif arg is None or isinstance(arg, (list, tuple)):
        if XLA:
            device = xm.xla_device()
        else:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(arg, str):
        if arg == 'xla' and XLA:
            device = xm.xla_device()
        else:
            device = torch.device(arg)
    
    if isinstance(arg, (list, tuple)):
        if isinstance(arg[0], int):
            device_ids = list(arg)
        elif isinstance(arg[0], str) and arg[0].isnumeric():
             device_ids = [ int(a) for a in arg ]
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


def set_random_seeds(random_seed=0, deterministic=False):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = deterministic
    random.seed(random_seed)


def get_gpu_memory():
    """
    Code borrowed from: 
    https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4

    Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def get_time(time_format='%H:%M:%S'):
    return time.strftime(time_format, time.localtime())
