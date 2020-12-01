import re
import sys
import time
from pathlib import Path
from tqdm import tqdm
from copy import copy, deepcopy

import numpy as np
import pandas as pd

import torch
import torch.utils.data as D
from .utils import * 
from .tensorboard import DummyTensorBoardLogger
from .temperature_scaling import TemperatureScaler
from .fp16util import network_to_half
from .callbacks import (
    CallbackEnv, TorchLogger, EarlyStopping)

try:
    from torchsummary import summary
except ModuleNotFoundError:
    print('torchsummary not found.')

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
except ModuleNotFoundError:
    print('torch_xla not found.')

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
    APEX_FLAG = True
except ModuleNotFoundError:
    print('nvidia apex not found.')
    APEX_FLAG = False


'''
Trainer
'''

class TorchTrainer:
    '''
    Simple Trainer for PyTorch models
    
    This is something similar to PyTorch Lightning, but this works with vanilla PyTorch modules.
    '''

    def __init__(self, 
                 model, device=None, fp16=False, xla=False, serial='Trainer'):
        
        self.serial = serial
        self.device = device
        self.on_xla = xla
        self.model = model

        if self.device is None:
            if self.on_xla:
                self.device = xm.xla_device()
            else:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Some hidden params
        self.apex_opt_level = 'O1'
        self.argument_index_to_model = [0]
        self.argument_index_to_metric = None

    def wrap_model(self):
        '''
        fp16 and data parallel
        '''
        if self.fp16:
            if APEX_FLAG:
                self.model, self.optimizer = amp.initialize(
                    self.model, self.optimizer,
                    opt_level=self.apex_opt_level, verbosity=0)
                self.logger('Model, Optimizer -> fp16 (apex)')
            else:
                self.model = network_to_half(self.model)
                self.logger('Model -> fp16 (simple)')

        if self.parallel == 'dp':
            if self.on_xla:
                raise NotImplementedError(
                    '[WIP] Parallel training on xla devices.')

            if torch.cuda.device_count() > 1:
                all_devices = list(range(torch.cuda.device_count()))
                if self.fp16 and APEX_FLAG:
                    self.model = nn.parallel.DataParallel(self.model)
                else:
                    self.model = nn.parallel.DataParallel(self.model)

                self.logger(
                    f'{torch.cuda.device_count()}({all_devices}) gpus found.')
        elif self.parallel == 'ddp':
            self.world_size = torch.cuda.device_count()
            raise NotImplementedError('[WIP] DDP.')
        elif self.parallel is not None:
            raise ValueError(f'Unknown type of parallel {self.parallel}')

    def train_one_epoch(self, loader, grad_accumulations=1):
        loss_total = 0.0
        total_batch = len(loader.dataset) / loader.batch_size
        approx = []
        target = []
        others = []

        self.model.train()
        for batch_i, inputs in enumerate(loader):
            batches_done = len(loader) * self.global_epoch + batch_i

            inputs = [t.to(self.device) for t in inputs]
            y = inputs[-1]  # !: the last input is always target
            _y = self.model(*[inputs[i] for i in self.argument_index_to_model])
            if self.fp16:
                _y = _y.float()

            approx.append(_y.clone().detach())
            target.append(y.clone().detach())
            if self.argument_index_to_metric is not None:
                others.append(
                    inputs[self.argument_index_to_metric].clone().detach())

            loss = self.criterion(_y, y)
            if self.fp16 and APEX_FLAG:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if batch_i == 0:
                # Save output dimension in the first run
                self.out_dim = _y.shape[1:]

            if (batch_i + 1) % grad_accumulations == 0:
                # Accumulates gradient before each step
                loss = loss / grad_accumulations  # normalize loss
                if self.on_xla:
                    xm.optimizer_step(self.optimizer, barrier=True)
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            for param_group in self.optimizer.param_groups:
                learning_rate = param_group['lr']
            log_train_batch = [
                (f'batch_loss_train[{self.serial}]', loss.item()),
                (f'batch_lr_train[{self.serial}]', learning_rate)
            ]
            self.tb_logger.list_of_scalars_summary(
                log_train_batch, batches_done)

            batch_weight = len(y) / loader.batch_size
            loss_total += loss.item() / total_batch * batch_weight

        approx = torch.cat(approx).cpu()
        target = torch.cat(target).cpu()
        if len(others) > 0:
            others = torch.cat(others).cpu()
        if self.eval_metric is None:
            metric_total = -loss_total
        else:
            if len(others) > 0:
                metric_total = self.eval_metric(approx, target, others)
            else:
                metric_total = self.eval_metric(approx, target)
        monitor_metrics_total = []
        for monitor_metric in self.monitor_metrics:
            if len(others) > 0:
                monitor_metrics_total.append(
                    monitor_metric(approx, target, others))
            else:
                monitor_metrics_total.append(monitor_metric(approx, target))

        log_train = [
            (f'epoch_metric_train[{self.serial}]', metric_total),
            (f'epoch_loss_train[{self.serial}]', loss_total)
        ]
        self.tb_logger.list_of_scalars_summary(log_train, self.global_epoch)
        self.evals_result['train']['loss'].append(loss_total)
        self.evals_result['train']['metric'].append(metric_total)

        return loss_total, metric_total, monitor_metrics_total

    def valid_one_epoch(self, loader):
        loss_total = 0.0
        total_batch = len(loader.dataset) / loader.batch_size
        approx = []
        target = []
        others = []

        self.model.eval()
        with torch.no_grad():
            for inputs in loader:
                inputs = [t.to(self.device) for t in inputs]
                y = inputs[-1] # !: the last input is always target
                _y = self.model(*[inputs[i] for i in self.argument_index_to_model])
                if self.fp16:
                    _y = _y.float()

                approx.append(_y.clone().detach())
                target.append(y.clone().detach())
                if self.argument_index_to_metric is not None:
                    others.append(inputs[self.argument_index_to_metric].clone().detach())

                loss = self.criterion(_y, y)

                batch_weight = len(y) / loader.batch_size
                loss_total += loss.item() / total_batch * batch_weight

        approx = torch.cat(approx).cpu()
        target = torch.cat(target).cpu()
        if len(others) > 0:
            others = torch.cat(others).cpu()
        if self.eval_metric is None:
            metric_total = -loss_total
        else:
            if len(others) > 0:
                metric_total = self.eval_metric(approx, target, others)
            else:
                metric_total = self.eval_metric(approx, target)
        monitor_metrics_total = []
        for monitor_metric in self.monitor_metrics:
            if len(others) > 0:
                monitor_metrics_total.append(monitor_metric(approx, target, others))
            else:
                monitor_metrics_total.append(monitor_metric(approx, target))

        log_valid = [
            (f'epoch_metric_valid[{self.serial}]', metric_total),
            (f'epoch_loss_valid[{self.serial}]', loss_total)
        ]
        self.tb_logger.list_of_scalars_summary(log_valid, self.global_epoch)
        self.evals_result['valid']['loss'].append(loss_total)
        self.evals_result['valid']['metric'].append(metric_total)

        return loss_total, metric_total, monitor_metrics_total

    def predict(self, loader, path=None, test_time_augmentations=1, verbose=True):
        if loader is None:
            self.logger('Skipping prediction...')
            return None
        prediction = []

        self.model.eval()
        with torch.no_grad():
            for inputs in loader:
                inputs = [t.to(self.device) for t in inputs]
                y = inputs[-1]  # !: the last input is always target
                if self.fp16 and APEX_FLAG:
                    with amp.disable_casts():
                        _y = self.model(*[inputs[i] for i in self.argument_index_to_model])
                else:
                    _y = self.model(*[inputs[i] for i in self.argument_index_to_model])
                prediction.append(_y.detach())
        
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
            loader_valid=None, loader_test=None, callbacks=[],
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
        self.eval_metric = eval_metric
        self.monitor_metrics = monitor_metrics
        self.logger = logger
        self.tb_logger = tb_logger
        self.callbacks = callbacks
        self.fp16 = fp16
        self.parallel = parallel

        self.best_epoch = 1
        self.best_score = None
        self.patience = 0
        self.global_epoch = 1
        self.stop_train = False
        self.checkpoint = False
        self.evals_result = {
            'train': {'loss': [], 'metric': []},
            'valid': {'loss': [], 'metric': []}
        }
        self.outoffold = None
        self.prediction = None
        

        ''' Configure loggers '''
        if self.logger is None:
            self.logger = TorchLogger('')
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


        ''' Configure snapshot '''
        if export_dir is None:
            export_dir = Path().cwd()
        elif isinstance(export_dir, str):
            export_dir = Path(export_dir).expanduser()
        assert len(export_dir.suffix) ==  0 # export_dir must be directory
        export_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = export_dir / 'snapshot.pt'


        ''' Configure model '''
        self.model.to(self.device)
        if resume:
            self.load_snapshot(snapshot_path, load_callbacks=~ignore_callbacks)
            self.logger(f'{snapshot_path} is loaded. Continuing from epoch {self.global_epoch}.')
        else:
            if snapshot_path.exists():
                snapshot_path.unlink()
        self.wrap_model()
        

        ''' Train '''
        self.logger(f'Model is on {self.device}')
        self.max_epochs = self.global_epoch + num_epochs - 1
        loss_valid, metric_valid = np.inf, -np.inf

        for epoch in range(num_epochs):
            start_time = time.time()
            if self.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                if loss_valid is None: # no validation set
                    self.scheduler.step(np.inf)
                else:
                    self.scheduler.step(loss_valid)
            else:
                self.scheduler.step()

            ''' Training set '''
            loss_train, metric_train, monitor_metrics_train = \
                self.train_one_epoch(loader, grad_accumulations)

            if loader_valid is None:
                ''' No validation set '''
                loss_valid, metric_valid, monitor_metrics_valid = None, None, None

            else:
                ''' Validation set '''
                loss_valid, metric_valid, monitor_metrics_valid = \
                    self.valid_one_epoch(loader_valid)

            ''' Callbacks '''
            results = {
                'train_loss': loss_train,
                'train_metric': metric_train,
                'train_monitor': monitor_metrics_train,
                'valid_loss': loss_valid,
                'valid_metric': metric_valid,
                'valid_monitor': monitor_metrics_valid,
                'patience': self.patience
            }
            for func in self.callbacks + [self.logger._callback]:
                func(CallbackEnv(self, epoch, results))

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
