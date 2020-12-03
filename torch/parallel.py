'''
WIP
'''
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.dataloader import _InfiniteConstantSampler
from torch.utils.data.distributed import DistributedSampler

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel
    APEX = True
except:
    from torch.nn.parallel import DistributedDataParallel
    APEX = False


DDP_TMP_CHECKPOINT = Path('.ddp.tmp')


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = False
    random.seed(random_seed)


def _train_one_epoch_ddp(
        rank, world_size,
        model, optimizer, scheduler, loader,
        criterion, eval_metric, monitor_metrics,
        argument_target=-1, argument_to_model=[0], 
        argument_to_metric=None, argument_to_criterion=None,
        batch_scheduler=False):

    ''' Transfer models etc '''
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    model.to(rank)
    if APEX:
        model = DistributedDataParallel(
            model, delay_allreduce=True)
    else:
        model = DistributedDataParallel(
            model, device_ids=[rank], find_unused_parameters=True)
    
    ''' Swap loader '''
    if isinstance(
        loader.sampler, 
        (RandomSampler, SequentialSampler, _InfiniteConstantSampler)):
        loader.__initialized = False
        loader.sampler = DistributedSampler(loader.dataset)
        loader.num_workers = 0
        loader.__initialized = True

    
    loss_total = 0.0
    total_batch = len(loader.dataset) / loader.batch_size
    approxs = []
    targets = []
    appendices = []

    ddp_model.train()
    for batch_i, inputs in enumerate(loader):
        inputs = [t.to(rank) for t in inputs]
        target = inputs[argument_target]
        approx = model(*[inputs[i] for i in argument_to_model])

        approxs.append(approx.clone().detach())
        targets.append(target.clone().detach())
        if argument_to_metric is not None:
            appendices.append(inputs[argument_to_metric].clone().detach())

        if argument_to_criterion is not None:
            loss = criterion(
                approx, target, inputs[argument_to_criterion])
        else:
            loss = criterion(approx, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch_scheduler:
            scheduler.step()

        if rank == 0:
            batch_weight = len(target) / loader.batch_size
            loss_total += loss.item() / total_batch * batch_weight

    if rank == 0:
        approxs = torch.cat(approxs).cpu()
        targets = torch.cat(targets).cpu()
        if len(appendices) > 0:
            appendices = torch.cat(appendices).cpu()

        if eval_metric is None:
            metric_total = -loss_total
        else:
            if len(appendices) > 0:
                metric_total = eval_metric(approxs, targets, appendices)
            else:
                metric_total = eval_metric(approxs, targets)

        monitor_metrics_total = []
        for monitor_metric in monitor_metrics:
            if len(appendices) > 0:
                monitor_metrics_total.append(
                    monitor_metric(approxs, targets, appendices))
            else:
                monitor_metrics_total.append(
                    monitor_metric(approxs, targets))
        
        print(f'LOG: {len(approxs)}')

        torch.save({
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 
            'loss': loss_total,
            'metric': metric_total,
            'monitor': monitor_metrics_total,
        }, str(DDP_TMP_CHECKPOINT))

    dist.destroy_process_group()
