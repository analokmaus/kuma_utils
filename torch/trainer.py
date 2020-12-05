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

from .utils import XLA, get_device
from .tensorboard import DummyTensorBoardLogger
from .temperature_scaling import TemperatureScaler
from .callbacks import (
    CallbackEnv, TorchLogger, EarlyStopping)
from .hooks import SimpleHook
from .parallel import _train_ddp_worker

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
        self.device, self.device_ids = get_device(device)
        self.xla = self.device.type == 'xla'
        self.model = model

        # Some implicit attributes
        self.apex_opt_level = 'O1'
        self.scheduler_target = None
        self.amp_backend = 'AMP'
        self.progress_bar = False

    def _register_callbacks(self, callbacks):
        self.before_train = [func.before_train for func in callbacks]
        self.after_train = [func.after_train for func in callbacks]

    def _register_hook(self, hook):
        self.batch_train = hook.batch_train
        self.batch_test = hook.batch_test
        self.epoch_eval = hook.epoch_eval

    def _configure_model(self):
        ''' Mixed precision '''
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

        ''' Parallel training '''
        if self.parallel == 'dp':
            if self.xla:
                raise NotImplementedError(
                    'Data Parallel training on xla devices is not supported.')
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
        loss_total = .0
        total_batch = len(loader.dataset) / loader.batch_size
        approxs = []
        targets = []
        extras = []
        if self.amp_backend == 'AMP' and self.fp16:
            scaler = amp.GradScaler()

        self.model.train()
        if self.progress_bar:
            iterator = enumerate(tqdm(loader))
        else:
            iterator = enumerate(loader)

        for batch_i, inputs in iterator:
            self.optimizer.zero_grad()
            batches_done = len(loader) * (self.global_epoch-1) + batch_i
            inputs = [t.to(self.device) for t in inputs]

            if self.amp_backend == 'AMP' and self.fp16:
                with amp.autocast():
                    approx, target, loss, metric, extra = self.batch_train(self, inputs)
                approx = approx.float()
                loss = loss / grad_accumulations
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                
            else:
                approx, target, loss, metric, extra = self.batch_train(self, inputs)
                loss = loss / grad_accumulations
                if self.amp_backend == 'APEX' and self.fp16:
                    approx = approx.float()
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

            approxs.append(approx.clone().detach())
            targets.append(target.clone().detach())
            if extra is not None:
                extras.append(extra.clone().detach())

            if batch_i == 0:
                # Save output dimension in the first run
                self.out_dim = approx.shape[1:]

            if (batch_i + 1) % grad_accumulations == 0:
                if self.xla:
                    xm.optimizer_step(self.optimizer, barrier=True)
                else:
                    self.optimizer.step()
                if self.batch_scheduler:
                    self.scheduler.step()

            ''' TensorBoard logging '''
            learning_rate = [ param_group['lr'] \
                for param_group in self.optimizer.param_groups ]
            logs = [
                ('batch_train_loss', loss.item()),
                ('batch_train_lr', learning_rate)
            ]
            if metric is not None:
                logs.append(('batch_train_mertric', metric))
            self.tb_logger.list_of_scalars_summary(logs, batches_done)

            batch_weight = len(target) / loader.batch_size
            loss_total += loss.item() / total_batch * batch_weight

        approxs = torch.cat(approxs).cpu()
        targets = torch.cat(targets).cpu()
        if len(extras) > 0:
            extras = torch.cat(extras).cpu()
        
        metric_total, monitor_metrics_total = \
            self.epoch_eval(self, approxs, targets, extras)

        ''' TensorBoard logging '''
        logs = [
            ('epoch_train_loss', loss_total), 
            ('epoch_train_metric', metric_total),
        ]
        self.tb_logger.list_of_scalars_summary(logs, self.global_epoch)

        self.evals_result['train']['loss'].append(loss_total)
        self.evals_result['train']['metric'].append(metric_total)
        return loss_total, metric_total, monitor_metrics_total

    def _valid_one_epoch(self, loader):
        loss_total = .0
        total_batch = len(loader.dataset) / loader.batch_size
        approxs = []
        targets = []
        extras = []

        self.model.eval()
        if self.progress_bar:
            iterator = enumerate(tqdm(loader))
        else:
            iterator = enumerate(loader)
        
        with torch.no_grad():
            for batch_i, inputs in iterator:
                batches_done = len(loader) * (self.global_epoch-1) + batch_i
                inputs = [t.to(self.device) for t in inputs]

                if self.amp_backend == 'APEX' and self.fp16:
                    with amp.disable_casts():  # inference should be in FP32
                        approx, target, loss, metric, extra = self.batch_train(self, inputs)

                else:
                    approx, target, loss, metric, extra = self.batch_train(self, inputs)

                approxs.append(approx.clone().detach())
                targets.append(target.clone().detach())
                if extra is not None:
                    extras.append(extra.clone().detach())

                ''' TensorBoard logging '''
                logs = [
                    ('batch_valid_loss', loss.item()),
                ]
                if metric is not None:
                    logs.append(('batch_valid_mertric', metric))
                self.tb_logger.list_of_scalars_summary(logs, batches_done)

                batch_weight = len(target) / loader.batch_size
                loss_total += loss.item() / total_batch * batch_weight

        approxs = torch.cat(approxs).cpu()
        targets = torch.cat(targets).cpu()
        if len(extras) > 0:
            extras = torch.cat(extras).cpu()

        metric_total, monitor_metrics_total = \
            self.epoch_eval(self, approxs, targets, extras)

        ''' TensorBoard logging '''
        logs = [
            ('epoch_valid_loss', loss_total), 
            ('epoch_valid_metric', metric_total),
        ]
        self.tb_logger.list_of_scalars_summary(logs, self.global_epoch)

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
                self, loader, loader_valid, num_epochs)
        )
        
    def predict(self, loader, path=None, test_time_augmentations=1, verbose=True):
        if loader is None:
            self.logger('Loader is None. Nothing to predict.')
            return None

        prediction = []

        self.model.eval()
        with torch.no_grad():
            for inputs in loader:
                inputs = [t.to(self.device) for t in inputs]
                if self.amp_backend == 'APEX' and self.fp16:
                    with amp.disable_casts(): # inference should be in FP32
                        approx = self.batch_test(self, inputs)
                else:
                    approx = self.batch_test(self, inputs)
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
        }, path)

    def load_snapshot(self, path, device=None, 
                      load_epoch=True, load_scheduler=True):
        if device is None:
            device = self.device
        checkpoint = torch.load(path, map_location=device)
        if isinstance(
                self.model, 
                (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            self.model.module.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if load_scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        if load_epoch:
            self.global_epoch = checkpoint['global_epoch']

    def train(self,
            # Essential
            criterion, optimizer, scheduler, loader, num_epochs, 
            loader_valid=None, loader_test=None, batch_scheduler=False, 
            hook=SimpleHook(), callbacks=[],
            # Snapshot
            export_dir=None, resume=False, 
            # Special training
            fp16=False, parallel=None, 
            grad_accumulations=1, calibrate_model=False,
            # Evaluation
            eval_metric=None, monitor_metrics=[],
            # Prediction
            test_time_augmentations=1, predict_valid=True, predict_test=True, 
            # Logger
            logger=None, tb_logger=None
        ):
        # Register params
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_scheduler = batch_scheduler
        self.eval_metric = eval_metric
        self.monitor_metrics = monitor_metrics
        self.logger = logger
        self.tb_logger = tb_logger
        self.fp16 = fp16
        self.parallel = parallel
        self._register_callbacks(callbacks)
        self._register_hook(hook)
        # Important flags
        self.global_epoch = 1
        self.stop_train = False
        self.checkpoint = False
        # Results
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
        if not isinstance(self.monitor_metrics, (list, tuple)):
            self.monitor_metrics = [self.monitor_metrics]

        ''' Configure model '''
        self.model.to(self.device)
        if resume:
            self.load_snapshot(snapshot_path)
            self.logger(f'{snapshot_path} is loaded. Continuing from epoch {self.global_epoch}.')
        else:
            if snapshot_path.exists():
                snapshot_path.unlink()
        self._configure_model()
        
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
                'train_loss': None,
                'train_metric': None,
                'train_monitor': None,
                'valid_loss': None,
                'valid_metric': None,
                'valid_monitor': None,
                'patience': self.patience,
                'learning_rate': [group['lr'] for group in self.optimizer.param_groups]
            }

            for epoch in range(num_epochs):
                ''' before train callbacks '''
                for func in self.before_train:
                    func(CallbackEnv(self, epoch, state))

                ''' Training set '''
                loss_train, metric_train, monitor_metrics_train = \
                        self._train_one_epoch(loader, grad_accumulations)

                ''' Validation set '''
                if loader_valid is None:
                    loss_valid, metric_valid, monitor_metrics_valid = None, None, None
                else:
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

                ''' After train callbacks '''
                for func in self.after_train + [self.logger._callback]:
                    func(CallbackEnv(self, epoch, state))

                if self.checkpoint:
                    ''' Save model '''
                    self.save_snapshot(snapshot_path)
                    self.checkpoint = False

                if self.stop_train:
                    ''' Early stop '''
                    self.logger('Training stopped by overfit detector.')
                    break

                self.global_epoch += 1
            else:
                ''' No early stop till the end '''
                self.save_snapshot(snapshot_path)

        ''' Prediction '''
        self.logger(
            f'Best epoch is [{self.best_epoch}], best score is [{self.best_score}].')
        if snapshot_path.exists():
            self.load_snapshot(
                str(snapshot_path), load_epoch=False, load_scheduler=False)
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

    def __repr__(self):
        print_items = [
            'device', 'device_ids', 
            'optimizer', 'scheduler', 'criterion', 'eval_metric', 'monitor_metrics', 
            'before_train', 'after_train', 'logger'
        ]
        print_text = f'TorchTrainer({self.serial})\n'
        for item in print_items:
            try:
                print_text += f'{item}: {getattr(self, item)}\n'
            except:
                pass
        return print_text
