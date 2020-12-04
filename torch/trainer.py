import time
import os
import sys
from pathlib import Path
from tqdm import tqdm
from copy import copy, deepcopy

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from .utils import * 
from .tensorboard import DummyTensorBoardLogger
from .temperature_scaling import TemperatureScaler
from .callbacks import (
    CallbackEnv, TorchLogger, EarlyStopping)
from .parallel import _train_ddp_worker

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    XLA = True
except ModuleNotFoundError:
    XLA = False

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


class TorchTrainer:
    '''
    Simple Trainer for PyTorch models
    
    This is something similar to PyTorch Lightning, but this works with vanilla PyTorch modules.
    '''

    def __init__(self, 
                 model, device=None, serial='Trainer'):
        
        self.serial = serial
        self.device, self.device_ids = self._get_device(device)
        self.model = model

        # Some hidden params
        self.apex_opt_level = 'O1'
        self.argument_to_model = [0]
        self.argument_target = -1
        self.argument_to_criterion = None
        self.argument_to_metric = None
        self.scheduler_target = None
        self.amp_backend = 'AMP'
        self.progress_bar = False

    def _get_device(self, device):
        self.on_xla = False
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
                device_count = torch.cuda.device_count()
                if device_count > 1:
                    device_ids = list(range(device_count))
                else:
                    device_ids = [0]
            else:
                device = torch.device('cpu')
                device_ids = None

        elif isinstance(device, str):
            if 'xla' in device:
                assert XLA
                self.on_xla = True
                if device in xm.get_xla_supported_devices():
                    device = xm.xla_device(device.split(':')[-1])
                else:
                    device = xm.xla_device()
            else:
                device = torch.device(device)

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
            elif device.type == 'xla':
                device_ids = [device.index]
            elif device.type == 'cpu':
                device_ids = None

        elif isinstance(device, (list, tuple)):
            if torch.cuda.is_available():
                device = torch.device('cuda')
                device_ids = device
            else:
                raise ValueError('Cuda is not availbale.')
        
        return device, device_ids

    def _wrap_model(self):
        '''
        fp16 and parallel
        '''
        if self.fp16:
            if self.amp_backend == 'APEX' and APEX:
                self.model, self.optimizer = amp.initialize(
                    self.model, self.optimizer,
                    opt_level=self.apex_opt_level, verbosity=0)
                self.logger('Mixed precision training on apex.')
            elif AMP:
                self.logger('Mixed precision training on torch amp.')
            else:
                self.fp16 = False
                self.logger('No mixed precision training backend found. fp16 is set False.')

        if self.parallel == 'dp':
            if self.on_xla:
                raise NotImplementedError(
                    '[WIP] Data Parallel training on xla devices.')
            else:
                self.model = nn.parallel.DataParallel(
                    self.model, device_ids=self.device_ids)
                self.logger(f'DataParallel on devices {self.device_ids}')

        elif self.parallel == 'ddp':
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            self.world_size = len(self.device_ids)
            self.logger(f'DistributedDataParallel on devices {self.device_ids}')

        elif self.parallel is not None:
            raise ValueError(f'Unknown type of parallel {self.parallel}')

    def _train_one_epoch(self, loader, grad_accumulations=1):
        loss_total = 0.0
        total_batch = len(loader.dataset) / loader.batch_size
        approxs = []
        targets = []
        appendices = []
        if self.amp_backend == 'AMP' and self.fp16:
            scaler = amp.GradScaler()

        self.model.train()
        if self.progress_bar:
            iterator = enumerate(tqdm(loader))
        else:
            iterator = enumerate(loader)
        for batch_i, inputs in iterator:
            self.optimizer.zero_grad()

            batches_done = len(loader) * self.global_epoch + batch_i
            inputs = [t.to(self.device) for t in inputs]
            target = inputs[self.argument_target]

            if self.amp_backend == 'AMP' and self.fp16:
                with amp.autocast():
                    approx = self.model(*[inputs[i]
                                          for i in self.argument_to_model])
                    if self.argument_to_criterion is not None:
                        loss = self.criterion(
                            approx, target, inputs[self.argument_to_criterion])
                    else:
                        loss = self.criterion(approx, target)
                approx = approx.float()
                loss = loss / grad_accumulations
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                
            else:
                approx = self.model(*[inputs[i] for i in self.argument_to_model])
                if self.argument_to_criterion is not None:
                    loss = self.criterion(
                        approx, target, inputs[self.argument_to_criterion])
                else:
                    loss = self.criterion(approx, target)
                loss = loss / grad_accumulations
                if self.amp_backend == 'APEX' and self.fp16:
                    approx = approx.float()
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

            approxs.append(approx.clone().detach())
            targets.append(target.clone().detach())
            if self.argument_to_metric is not None:
                appendices.append(
                    inputs[self.argument_to_metric].clone().detach())

            if batch_i == 0:
                # Save output dimension in the first run
                self.out_dim = approx.shape[1:]

            if (batch_i + 1) % grad_accumulations == 0:
                if self.on_xla:
                    xm.optimizer_step(self.optimizer, barrier=True)
                else:
                    self.optimizer.step()
                if self.batch_scheduler:
                    self.scheduler.step()

            for param_group in self.optimizer.param_groups:
                learning_rate = param_group['lr']
            log_train_batch = [
                (f'batch_loss_train[{self.serial}]', loss.item()),
                (f'batch_lr_train[{self.serial}]', learning_rate)
            ]
            self.tb_logger.list_of_scalars_summary(
                log_train_batch, batches_done)

            batch_weight = len(target) / loader.batch_size
            loss_total += loss.item() / total_batch * batch_weight

        approxs = torch.cat(approxs).cpu()
        targets = torch.cat(targets).cpu()
        if len(appendices) > 0:
            appendices = torch.cat(appendices).cpu()
        
        if self.eval_metric is None:
            metric_total = -loss_total
        else:
            if len(appendices) > 0:
                metric_total = self.eval_metric(approxs, targets, appendices)
            else:
                metric_total = self.eval_metric(approxs, targets)
        
        monitor_metrics_total = []
        for monitor_metric in self.monitor_metrics:
            if len(appendices) > 0:
                monitor_metrics_total.append(
                    monitor_metric(approxs, targets, appendices))
            else:
                monitor_metrics_total.append(
                    monitor_metric(approxs, targets))

        log_train = [
            (f'epoch_metric_train[{self.serial}]', metric_total),
            (f'epoch_loss_train[{self.serial}]', loss_total)
        ]
        self.tb_logger.list_of_scalars_summary(log_train, self.global_epoch)
        self.evals_result['train']['loss'].append(loss_total)
        self.evals_result['train']['metric'].append(metric_total)

        return loss_total, metric_total, monitor_metrics_total

    def _valid_one_epoch(self, loader):
        loss_total = 0.0
        total_batch = len(loader.dataset) / loader.batch_size
        approxs = []
        targets = []
        appendices = []

        self.model.eval()
        with torch.no_grad():
            for batch_i, inputs in enumerate(loader):
                batches_done = len(loader) * self.global_epoch + batch_i

                inputs = [t.to(self.device) for t in inputs]
                target = inputs[self.argument_target]
                if self.amp_backend == 'APEX' and self.fp16:
                    with amp.disable_casts():
                        approx = self.model(*[inputs[i] for i in self.argument_to_model])
                else:
                    approx = self.model(*[inputs[i] for i in self.argument_to_model])

                if self.argument_to_criterion is not None:
                    loss = self.criterion(
                        approx, target, inputs[self.argument_to_criterion])
                else:
                    loss = self.criterion(approx, target)

                approxs.append(approx.clone().detach())
                targets.append(target.clone().detach())
                if self.argument_to_metric is not None:
                    appendices.append(
                        inputs[self.argument_to_metric].clone().detach())

                batch_weight = len(target) / loader.batch_size
                loss_total += loss.item() / total_batch * batch_weight

        approxs = torch.cat(approxs).cpu()
        targets = torch.cat(targets).cpu()
        if len(appendices) > 0:
            appendices = torch.cat(appendices).cpu()

        if self.eval_metric is None:
            metric_total = -loss_total
        else:
            if len(appendices) > 0:
                metric_total = self.eval_metric(approxs, targets, appendices)
            else:
                metric_total = self.eval_metric(approxs, targets)

        monitor_metrics_total = []
        for monitor_metric in self.monitor_metrics:
            if len(appendices) > 0:
                monitor_metrics_total.append(
                    monitor_metric(approxs, targets, appendices))
            else:
                monitor_metrics_total.append(
                    monitor_metric(approxs, targets))

        log_valid = [
            (f'epoch_metric_valid[{self.serial}]', metric_total),
            (f'epoch_loss_valid[{self.serial}]', loss_total)
        ]
        self.tb_logger.list_of_scalars_summary(log_valid, self.global_epoch)
        self.evals_result['valid']['loss'].append(loss_total)
        self.evals_result['valid']['metric'].append(metric_total)

        return loss_total, metric_total, monitor_metrics_total

    def _train_ddp(self, loader, loader_valid, num_epochs):
        '''
        Experimental DDP implementation
        '''
        mp.spawn(
            _train_ddp_worker,
            nprocs=self.world_size,
            args=(
                self.world_size,
                self.model, self.optimizer, self.scheduler, 
                loader, loader_valid, 
                self.criterion, self.eval_metric, self.monitor_metrics,
                num_epochs, self.global_epoch, self.logger, self.callbacks, self.patience, 
                self.scheduler_target, 
                self.argument_target, self.argument_to_model,
                self.argument_to_metric, self.argument_to_criterion,
                self.batch_scheduler, self.fp16
            )
        )
        
    def predict(self, loader, path=None, test_time_augmentations=1, verbose=True):
        if loader is None:
            self.logger('Skipping prediction...')
            return None
        prediction = []

        self.model.eval()
        with torch.no_grad():
            for inputs in loader:
                inputs = [t.to(self.device) for t in inputs]
                target = inputs[self.argument_target]
                if self.amp_backend == 'APEX' and self.fp16:
                    with amp.disable_casts():
                        approx = self.model(*[inputs[i]
                                              for i in self.argument_to_model])
                else:
                    approx = self.model(*[inputs[i]
                                          for i in self.argument_to_model])
                prediction.append(approx.detach())
        
        prediction = torch.cat(prediction).cpu().numpy()

        if path is not None:
            np.save(path, prediction)
        self.logger(f'Prediction done. exported to {path}')

        return prediction

    def save_snapshot(self, path):
        if isinstance(self.model, torch.nn.DataParallel):
            module = self.model.module
        else:
            module = self.model

        torch.save({
            'global_epoch': self.global_epoch,
            'model': module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'callbacks': [func.state_dict() for func in self.callbacks],
        }, path)

    def load_snapshot(self, path, 
                      load_epoch=True, load_scheduler=True, load_callbacks=True):
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if load_scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        if load_callbacks:
            for i in range(len(self.callbacks)):
                self.callbacks[i].load_state_dict(checkpoint['callbacks'][i])
        if load_epoch:
            self.global_epoch = checkpoint['global_epoch']

    def train(self,
            # Essential
            criterion, optimizer, scheduler, loader, num_epochs, 
            loader_valid=None, loader_test=None, batch_scheduler=False, callbacks=[],
            # Snapshot
            export_dir=None, resume=False, ignore_callbacks=False, 
            # Special training
            fp16=False, parallel=None, 
            grad_accumulations=1, calibrate_model=False,
            # Evaluation
            eval_metric=None, monitor_metrics=[],
            # Prediction
            test_time_augmentations=1, predict_valid=True, predict_test=True,  # Prediction
            # Logger
            logger=None, tb_logger=None
        ):

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_scheduler = batch_scheduler
        self.eval_metric = eval_metric
        self.monitor_metrics = monitor_metrics
        self.logger = logger
        self.tb_logger = tb_logger
        self.callbacks = callbacks
        self.fp16 = fp16
        self.parallel = parallel

        self.global_epoch = 1
        self.stop_train = False
        self.checkpoint = False
        self.evals_result = {
            'train': {'loss': [], 'metric': []},
            'valid': {'loss': [], 'metric': []}
        }
        self.outoffold = None
        self.prediction = None
        
        ''' Configure directory '''
        if export_dir is None:
            export_dir = Path().cwd()
        elif isinstance(export_dir, str):
            export_dir = Path(export_dir).expanduser()
        assert len(export_dir.suffix) == 0  # export_dir must be directory
        export_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = export_dir / 'snapshot.pt'

        ''' Configure loggers '''
        if self.logger is None:
            self.logger = TorchLogger(export_dir / 'log.log')
        elif isinstance(self.logger, (str, Path)):
            self.logger = TorchLogger(self.logger, file=True)
        elif isinstance(self.logger, TorchLogger):
            pass
        else:
            raise ValueError('Invalid type of logger.')

        if self.tb_logger is None:
            self.tb_logger = DummyTensorBoardLogger()

        ''' Configure loss function and metrics '''
        if eval_metric is None:
            self.logger(
                'eval_metric is not set. Inversed criterion will be used instead.')
        if not isinstance(self.monitor_metrics, (list, tuple, set)):
            self.monitor_metrics = [self.monitor_metrics]

        ''' Configure model '''
        self.model.to(self.device)
        if resume:
            self.load_snapshot(snapshot_path, load_callbacks=~ignore_callbacks)
            self.logger(f'{snapshot_path} is loaded. Continuing from epoch {self.global_epoch}.')
        else:
            if snapshot_path.exists():
                snapshot_path.unlink()
        self._wrap_model()
        
        ''' Train '''
        self.logger(f'Model is on {self.device}')
        self.max_epochs = self.global_epoch + num_epochs - 1
        self.best_epoch = self.global_epoch
        self.best_score = None
        self.patience = 0

        if self.parallel == 'ddp':
            self._train_ddp(loader, loader_valid, num_epochs)

        else:
            state = {
                'train_loss': np.inf,
                'train_metric': None,
                'train_monitor': None,
                'valid_loss': np.inf,
                'valid_metric': None,
                'valid_monitor': None,
                'patience': self.patience,
                'learning_rate': [group['lr'] for group in self.optimizer.param_groups]
            }

            for epoch in range(num_epochs):
                start_time = time.time()

                ''' Training set '''
                loss_train, metric_train, monitor_metrics_train = \
                        self._train_one_epoch(loader, grad_accumulations)

                if loader_valid is None:
                    ''' No validation set '''
                    loss_valid, metric_valid, monitor_metrics_valid = None, None, None
                else:
                    ''' Validation set '''
                    loss_valid, metric_valid, monitor_metrics_valid = \
                        self._valid_one_epoch(loader_valid)

                state.update({
                    'train_loss': loss_train,
                    'train_metric': metric_train,
                    'train_monitor': monitor_metrics_train,
                    'valid_loss': loss_valid,
                    'valid_metric': metric_valid,
                    'valid_monitor': monitor_metrics_valid,
                    'patience': self.patience,
                    'learning_rate': [group['lr'] for group in self.optimizer.param_groups]
                })

                if not self.batch_scheduler: # Epoch scheduler
                    if self.scheduler_target is not None:
                        self.scheduler.step(state[self.scheduler_target])
                    else:
                        self.scheduler.step()

                ''' Callbacks '''
                for func in self.callbacks + [self.logger._callback]:
                    func(CallbackEnv(self, epoch, state))

                if self.checkpoint:
                    ''' Save model '''
                    self.save_snapshot(snapshot_path)
                    self.checkpoint = False

                if self.stop_train:
                    ''' Early stop '''
                    self.logger('Training stopped by overfit detector.')
                    break

                ''' Not stopped '''
                self.global_epoch += 1
            else:
                ''' No early stop till the end '''
                self.save_snapshot(snapshot_path)

        ''' Prediction '''
        self.logger(
            f'Best epoch is [{self.best_epoch}], best score is [{self.best_score}].')
        if snapshot_path.exists():
            self.load_snapshot(
                str(snapshot_path), load_epoch=False, load_scheduler=False, load_callbacks=False)
        else:
            self.save_snapshot(snapshot_path)

        if calibrate_model:
            if loader_valid is None:
                raise ValueError(
                    'loader_valid is necessary for calibration.')
            else:
                self.calibrate_model(loader_valid)

        if predict_valid:
            if loader_valid is None:
                self.outoffold = self.predict(
                    loader, test_time_augmentations=test_time_augmentations)
            else:
                self.outoffold = self.predict(
                    loader_valid, test_time_augmentations=test_time_augmentations)
        if predict_test:
            self.prediction = self.predict(
                loader_test, test_time_augmentations=test_time_augmentations)
    
    fit = train # for compatibility

    def calibrate_model(self, loader):
        self.model = TemperatureScaler(self.model).to(self.device)
        self.model.set_temperature(loader)
