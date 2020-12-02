from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist


DDP_TMP_CHECKPOINT = Path('.ddp.tmp')


def _train_one_epoch_ddp(
        rank, world_size,
        model, optimizer, scheduler, loader,
        criterion, eval_metric, monitor_metrics,
        argument_target, argument_to_model, argument_to_metric, argument_to_criterion,
        batch_scheduler):

    dist.init_process_group(
        backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    model.to(rank)
    ddp_model = nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], find_unused_parameters=True)

    print(f'rank{rank}')
    loss_total = 0.0
    total_batch = len(loader.dataset) / loader.batch_size
    approxs = []
    targets = []
    appendices = []

    ddp_model.train()
    for batch_i, inputs in enumerate(loader):
        inputs = [t.to(rank) for t in inputs]
        target = inputs[argument_target]
        approx = ddp_model(*[inputs[i] for i in argument_to_model])

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

        torch.save({
            'model': ddp_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 
            'loss': loss_total,
            'metric': metric_total,
            'monitor': monitor_metrics_total,
        }, str(DDP_TMP_CHECKPOINT))

    dist.destroy_process_group()
