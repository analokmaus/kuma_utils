'''
WIP

TODO:
UGLY
'''
from pathlib import Path
import random
from dataclasses import dataclass, field
from typing import Any
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.dataloader import _InfiniteConstantSampler
from torch.utils.data.distributed import DistributedSampler

from .callbacks import CallbackEnv

try:
    from torch.cuda import amp
    AMP = True
    APEX = False
except ModuleNotFoundError:
    try:
        from apex import amp
        APEX = True
        AMP = True
    except ModuleNotFoundError:
        APEX = False
        AMP = False


DDP_TMP_CHECKPOINT = Path('.ddp.tmp')


def _set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = False
    random.seed(random_seed)


def _train_one_epoch_ddp(
        rank, model, loader, optimizer, scheduler,
        criterion, eval_metric, monitor_metrics,
        argument_target=-1, argument_to_model=[0],
        argument_to_metric=None, argument_to_criterion=None,
        batch_scheduler=False, use_amp=False):
    
    loss_total = 0.0
    total_batch = len(loader.dataset) / loader.batch_size
    approxs = []
    targets = []
    appendices = []
    if use_amp and AMP:
        scaler = amp.GradScaler()

    model.train()
    for batch_i, inputs in enumerate(loader):
        inputs = [t.to(rank) for t in inputs]
        target = inputs[argument_target]

        if use_amp and AMP:
            with amp.autocast():
                approx = model(*[inputs[i] for i in argument_to_model])
                if argument_to_criterion is not None:
                    loss = criterion(
                        approx, target, inputs[argument_to_criterion])
                else:
                    loss = criterion(approx, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            approx = model(*[inputs[i] for i in argument_to_model])
            if argument_to_criterion is not None:
                loss = criterion(
                    approx, target, inputs[argument_to_criterion])
            else:
                loss = criterion(approx, target)
            loss.backward()

        approxs.append(approx.clone().detach())
        targets.append(target.clone().detach())
        if argument_to_metric is not None:
            appendices.append(inputs[argument_to_metric].clone().detach())

        optimizer.step()
        optimizer.zero_grad()
        if batch_scheduler:
            scheduler.step()

        if rank == 0:
            batch_weight = len(target) / loader.batch_size
            loss_total += loss.item() / total_batch * batch_weight

    ''' Evaluate only on master device '''
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

        return loss_total, metric_total, monitor_metrics_total
    else:
        return None


def _valid_one_epoch_ddp(
        rank, model, loader, optimizer, scheduler,
        criterion, eval_metric, monitor_metrics,
        argument_target=-1, argument_to_model=[0],
        argument_to_metric=None, argument_to_criterion=None,
        batch_scheduler=False, use_amp=False):

    loss_total = 0.0
    total_batch = len(loader.dataset) / loader.batch_size
    approxs = []
    targets = []
    appendices = []

    model.eval()
    with torch.no_grad():
        for batch_i, inputs in enumerate(loader):
            inputs = [t.to(rank) for t in inputs]
            target = inputs[argument_target]

            approx = model(*[inputs[i] for i in argument_to_model])
            if argument_to_criterion is not None:
                loss = criterion(
                    approx, target, inputs[argument_to_criterion])
            else:
                loss = criterion(approx, target)

            approxs.append(approx.clone().detach())
            targets.append(target.clone().detach())
            if argument_to_metric is not None:
                appendices.append(inputs[argument_to_metric].clone().detach())

            if rank == 0:
                batch_weight = len(target) / loader.batch_size
                loss_total += loss.item() / total_batch * batch_weight

    ''' Evaluate only on master device '''
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

        return loss_total, metric_total, monitor_metrics_total
    else:
        return None


def _train_ddp_worker(
        rank, world_size, trainer, loader, loader_valid, num_epochs):

    ''' Transfer models etc '''
    _set_random_seeds(0)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    trainer.model.to(rank)
    trainer.model = DistributedDataParallel(
        trainer.model, device_ids=[rank], find_unused_parameters=True)
    
    ''' Swap loader '''
    for _loader in [loader, loader_test]:
        if _loader is None:
            continue
        if isinstance(
            _loader.sampler, 
            (RandomSampler, SequentialSampler, _InfiniteConstantSampler)):
            _loader._DataLoader__initialized = False
            _loader.sampler = DistributedSampler(loader.dataset)
            # _loader.num_workers = 0
            _loader_DataLoader__initialized = True

    ''' Training '''
    state = {
        'train_loss': None,
        'train_metric': None,
        'train_monitor': None,
        'valid_loss': None,
        'valid_metric': None,
        'valid_monitor': None,
        'patience': trainer.patience,
        'learning_rate': [group['lr'] for group in trainer.optimizer.param_groups]
    }

    for epoch in range(num_epochs):
        if rank == 0:
            ''' Before train callbacks '''
            for func in trainer.before_train:
                func(CallbackEnv(trainer, epoch, state))

        ''' Training set '''
        res_train = _train_one_epoch_ddp(
            rank, trainer.model, loader, trainer.optimizer, trainer.scheduler,
            trainer.criterion, trainer.eval_metric, trainer.monitor_metrics,
            trainer.argument_target, trainer.argument_to_model,
            trainer.argument_to_metric, trainer.argument_to_criterion,
            trainer.batch_scheduler, trainer.fp16)
        if rank == 0:
            loss_train, metric_train, monitor_metrics_train = res_train

        ''' Validation set '''
        if loader_valid is None or rank != 0:
            loss_valid, metric_valid, monitor_metrics_valid = None, None, None
        else:
            res_valid = _valid_one_epoch_ddp(
                rank, trainer.model, loader, trainer.optimizer, trainer.scheduler,
                trainer.criterion, trainer.eval_metric, trainer.monitor_metrics,
                trainer.argument_target, trainer.argument_to_model,
                trainer.argument_to_metric, trainer.argument_to_criterion,
                trainer.batch_scheduler, trainer.fp16)
            loss_valid, metric_valid, monitor_metrics_valid = res_valid
        
        ''' Callbacks '''
        if rank == 0:
            state.update({
                'train_loss': loss_train,
                'train_metric': metric_train,
                'train_monitor': monitor_metrics_train,
                'valid_loss': loss_valid,
                'valid_metric': metric_valid,
                'valid_monitor': monitor_metrics_valid,
                'patience': trainer.patience,
                'learning_rate': [group['lr'] for group in trainer.optimizer.param_groups]
            })
            
            if not traine.batch_scheduler:  # Epoch scheduler
                if trainer.scheduler_target is not None:
                    trainer.scheduler.step(state[scheduler_target])
                else:
                    trainer.scheduler.step()
            
            ''' After train callbacks '''
            for func in trainer.after_train + [trainer.logger._callback]:
                func(CallbackEnv(trainer, epoch, state))

            if trainer.checkpoint:
                ''' Save model '''
                trainer.save_snapshot(snapshot_path)
                trainer.checkpoint = False

            if trainer.stop_train:
                ''' Early stop '''
                logger('Training stopped by overfit detector.')
                break
            
            trainer.global_epoch += 1
    
    if rank == 0:
        logger('La fin')

    dist.destroy_process_group()
