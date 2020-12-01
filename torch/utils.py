import os
import torch
import torch.distributed as dist


def freeze_module(module):
    for i, param in enumerate(module.parameters()):
        param.requires_grad = False


def ddp_setup(rank, world_size):
    if sys.platform == 'win32':
        raise NotImplementedError('DDP for win32 is not implemented')
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # initialize the process group
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


def ddp_cleanup():
    dist.destroy_process_group()
