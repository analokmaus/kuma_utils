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
from .snapshot import *
from .logger import *
from .temperature_scaling import *
from .fp16util import network_to_half
from .stopper import DummyStopper
from .callback import CallbackEnv, DummyEvent

try:
    from torchsummary import summary
except ModuleNotFoundError:
    print('torch summary not found.')

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

    # Usage
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-3)
    NN_FIT_PARAMS = {
        'loader': loader_train,
        'loader_valid': loader_valid,
        'loader_test': loader_test,
        'criterion': nn.BCEWithLogitsLoss(),
        'optimizer': optimizer,
        'scheduler': StepLR(optimizer, step_size=10, gamma=0.9),
        'num_epochs': 100, 
        'stopper': EarlyStopping(patience=20, maximize=True),
        'logger': Logger('results/test/'), 
        'snapshot_path': Path('results/test/nn_best.pt'),
        'eval_metric': auc,
        'info_format': '[epoch] time data loss metric earlystopping',
        'info_train': False,
        'info_interval': 3
    }
    trainer = TorchTrainer(model, serial='test')
    trainer.fit(**NN_FIT_PARAMS)
    '''

    def __init__(self, 
                 model, device=None, serial='Trainer', 
                 fp16=False, xla=False):

        if device is None:
            if xla:
                device = xm.xla_device()
            else:
                device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')
        
        self.serial = serial
        self.device = device
        self.is_fp16 = fp16 # Automatically use apex if available
        self.is_xla = xla
        self.apex_opt_level = 'O1'
        self.model = model
        self.argument_index_to_model = [0]
        self.argument_index_to_metric = None
        print(f'[{self.serial}] On {self.device}.')

    def model_to_fp16(self):
        if APEX_FLAG:
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, 
                opt_level=self.apex_opt_level, verbosity=0)
            print(f'[{self.serial}] Model, Optimizer -> fp16 (apex)')
        else:
            self.model = network_to_half(self.model)
            print(f'[{self.serial}] Model -> fp16 (simple)')
        
    def model_to_parallel(self):
        if self.is_xla:
            print(
                f'[{self.serial}] Parallel training for xla devices is WIP.')

        if torch.cuda.device_count() > 1:
            all_devices = list(range(torch.cuda.device_count()))
            if self.is_fp16 and APEX_FLAG:
                self.model = nn.parallel.DataParallel(self.model)
            else:
                self.model = nn.parallel.DataParallel(self.model)

            print(f'[{self.serial}] {torch.cuda.device_count()}({all_devices}) gpus found.')

    def train_loop(self, loader, grad_accumulations=1, logger_interval=1):
        loss_total = 0.0
        total_batch = len(loader.dataset) / loader.batch_size
        approx = []
        target = []
        others = []

        self.model.train()
        for batch_i, inputs in enumerate(loader):
            batches_done = len(loader) * self.current_epoch + batch_i

            inputs = [t.to(self.device) for t in inputs]
            y = inputs[-1] # !: the last input is always target
            _y = self.model(*[inputs[i] for i in self.argument_index_to_model])
            if self.is_fp16:
                _y = _y.float()
            
            approx.append(_y.clone().detach())
            target.append(y.clone().detach())
            if self.argument_index_to_metric is not None:
                others.append(inputs[self.argument_index_to_metric].clone().detach())
            
            loss = self.criterion(_y, y)
            if self.is_fp16 and APEX_FLAG:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if batch_i == 0:
                # Save output dimension in the first run
                self.out_dim = _y.shape[1:]

            if (batch_i + 1) % grad_accumulations == 0:
                # Accumulates gradient before each step
                loss = loss / grad_accumulations # normalize loss
                if self.is_xla:
                    xm.optimizer_step(self.optimizer, barrier=True)
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                
            if batch_i % logger_interval == 0:
                for param_group in self.optimizer.param_groups:
                    learning_rate = param_group['lr']
                log_train_batch = [
                    (f'batch_loss_train[{self.serial}]', loss.item()),
                    (f'batch_lr_train[{self.serial}]', learning_rate)
                ]
                self.logger.list_of_scalars_summary(log_train_batch, batches_done)

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
        log_metrics_total = []
        for log_metric in self.log_metrics:
            if len(others) > 0:
                log_metrics_total.append(log_metric(approx, target, others))
            else:
                log_metrics_total.append(log_metric(approx, target))

        log_train = [
            (f'epoch_metric_train[{self.serial}]', metric_total),
            (f'epoch_loss_train[{self.serial}]', loss_total)
        ]
        self.logger.list_of_scalars_summary(log_train, self.current_epoch)
        self.log['train']['loss'].append(loss_total)
        self.log['train']['metric'].append(metric_total)
        
        return loss_total, metric_total, log_metrics_total

    def valid_loop(self, loader, grad_accumulations=1, logger_interval=1):
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
                if self.is_fp16:
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
        log_metrics_total = []
        for log_metric in self.log_metrics:
            if len(others) > 0:
                log_metrics_total.append(log_metric(approx, target, others))
            else:
                log_metrics_total.append(log_metric(approx, target))

        log_valid = [
            (f'epoch_metric_valid[{self.serial}]', metric_total),
            (f'epoch_loss_valid[{self.serial}]', loss_total)
        ]
        self.logger.list_of_scalars_summary(log_valid, self.current_epoch)
        self.log['valid']['loss'].append(loss_total)
        self.log['valid']['metric'].append(metric_total)

        return loss_total, metric_total, log_metrics_total

    def predict(self, loader, path=None, test_time_augmentations=1, verbose=True):
        if loader is None:
            print(f'[{self.serial}] No data to predict. Skipping prediction...')
            return None
        prediction = []

        self.model.eval()
        with torch.no_grad():
            for inputs in loader:
                inputs = [t.to(self.device) for t in inputs]
                y = inputs[-1]  # !: the last input is always target
                if self.is_fp16 and APEX_FLAG:
                    with amp.disable_casts():
                        _y = self.model(*[inputs[i] for i in self.argument_index_to_model])
                else:
                    _y = self.model(*[inputs[i] for i in self.argument_index_to_model])
                prediction.append(_y.detach())
        
        prediction = torch.cat(prediction).cpu().numpy()

        if path is not None:
            np.save(path, prediction)

        if verbose:
            print(f'[{self.serial}] Prediction done. exported to {path}')

        return prediction

    def print_info(self, info_items, info_seps, info):
        log_str = ''
        for sep, item in zip(info_seps, info_items):
            if item == 'time':
                current_time = time.strftime('%H:%M:%S', time.gmtime())
                log_str += current_time
            elif item == 'data':
                log_str += info[item]
            elif item in ['loss', 'metric']:
                log_str += f'{item}={info[item]:.{self.round_float}f}'
            elif item  == 'logmetrics':
                if len(info[item]) > 0:
                    for im, m in enumerate(info[item]): # list
                        log_str += f'{item}{im}={m:.{self.round_float}f}'
                        if im != len(info[item]) - 1:
                            log_str += ' '
            elif item == 'epoch':
                align = len(str(self.max_epochs))
                log_str += f'E{self.current_epoch:0{align}d}/{self.max_epochs}'
            elif item == 'earlystopping':
                if info['data'] == 'Trn':
                    continue
                counter, patience = self.stopper.state()
                best = self.stopper.score()
                if best is not None:
                    log_str += f'best={best:.{self.round_float}f}'
                    if counter > 0:
                        log_str += f'*({counter}/{patience})'
            log_str += sep
        if len(log_str) > 0:
            print(f'[{self.serial}] {log_str}')

    def train(self,
            # Essential
            criterion, optimizer, scheduler, 
            loader, num_epochs, loader_valid=None, loader_test=None,
            snapshot_path=None, resume=False,  # Snapshot
            multi_gpu=True, grad_accumulations=1, calibrate_model=False, # Train
            eval_metric=None, eval_interval=1, log_metrics=[], # Evaluation
            test_time_augmentations=1, predict_valid=True, predict_test=True,  # Prediction
            event=DummyEvent(), stopper=DummyStopper(),  # Train add-ons
            # Logger and info
            logger=DummyLogger(''), logger_interval=1, 
            info_train=True, info_valid=True, info_interval=1, round_float=6,
            info_format='[epoch] time data loss metric logmetrics earlystopping', verbose=True):

        if eval_metric is None:
            print(f'[{self.serial}] eval_metric is not set. Inversed criterion will be used instead.')

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.eval_metric = eval_metric
        self.log_metrics = log_metrics
        self.logger = logger
        self.event = deepcopy(event)
        self.stopper = deepcopy(stopper)
        self.current_epoch = 0

        self.log = {
            'train': {'loss': [], 'metric': []},
            'valid': {'loss': [], 'metric': []}
        }
        info_items = re.split(r'[^a-z]+', info_format)
        info_seps = re.split(r'[a-z]+', info_format)
        self.round_float = round_float

        if snapshot_path is None:
            snapshot_path = Path().cwd()
        if not isinstance(snapshot_path, Path):
            snapshot_path = Path(snapshot_path)
        if len(snapshot_path.suffix) > 0: # Is file
            self.root_path = snapshot_path.parent
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        else: # Is dir
            self.root_path = snapshot_path
            snapshot_path = snapshot_path/'snapshot.pt'
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)

        if not isinstance(self.log_metrics, (list, tuple, set)):
            self.log_metrics = [self.log_metrics]

        self.model.to(self.device)
        if resume:
            load_snapshots_to_model(
                snapshot_path, self.model, self.optimizer, self.scheduler, 
                self.stopper, self.event, device=self.device)
            self.current_epoch = load_epoch(snapshot_path)
            if verbose:
                print(
                    f'[{self.serial}] {snapshot_path} is loaded. Continuing from epoch {self.current_epoch}.')
        if self.is_fp16:
            self.model_to_fp16()
        if multi_gpu:
            self.model_to_parallel()

        self.max_epochs = self.current_epoch + num_epochs
        loss_valid, metric_valid = np.inf, -np.inf

        for epoch in range(num_epochs):
            self.current_epoch += 1
            start_time = time.time()
            if self.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                self.scheduler.step(loss_valid)
            else:
                self.scheduler.step()

            ''' Callback a priori '''
            self.event(
                CallbackEnv(
                    self.model, self.optimizer, self.scheduler, 
                    self.stopper, self.criterion, self.eval_metric, 
                    epoch, self.current_epoch, self.log
                )
            )

            ### Training
            loss_train, metric_train, log_metrics_train = \
                self.train_loop(loader, grad_accumulations, logger_interval)

            ### No validation set
            if loader_valid is None:
                early_stopping_target = metric_train

                if self.stopper(early_stopping_target):  # score improved
                    save_snapshots(snapshot_path, 
                                   self.current_epoch, self.model, 
                                   self.optimizer, self.scheduler, self.stopper, self.event)
                
                if info_train and epoch % info_interval == 0:
                    self.print_info(info_items, info_seps, {
                        'data': 'Trn',
                        'loss': loss_train,
                        'metric': metric_train, 
                        'logmetrics': log_metrics_train
                    })

                if self.stopper.stop():
                    if verbose:
                        print("[{}] Training stopped by overfit detector. ({}/{})".format(
                            self.serial, self.current_epoch-self.stopper.state()[1]+1, self.max_epochs))
                        print(f"[{self.serial}] Best score is {self.stopper.score():.{self.round_float}f}")
                    load_snapshots_to_model(str(snapshot_path), self.model, self.optimizer)
                    if predict_valid:
                        self.oof = self.predict(
                            loader, test_time_augmentations=test_time_augmentations, verbose=verbose)
                    if predict_test:
                        self.pred = self.predict(
                            loader_test, test_time_augmentations=test_time_augmentations, verbose=verbose)
                    break

                continue

            if info_train and epoch % info_interval == 0:
                self.print_info(info_items, info_seps, {
                    'data': 'Trn',
                    'loss': loss_train,
                    'metric': metric_train,
                    'logmetrics': log_metrics_train
                })

            ### Validation
            if epoch % eval_interval == 0:
                loss_valid, metric_valid, log_metrics_valid = \
                    self.valid_loop(loader_valid, grad_accumulations, logger_interval)
                
                early_stopping_target = metric_valid
                if self.stopper(early_stopping_target):  # score improved
                    save_snapshots(snapshot_path,
                                   self.current_epoch, self.model,
                                   self.optimizer, self.scheduler, self.stopper, self.event)

                if info_valid and epoch % info_interval == 0:
                    self.print_info(info_items, info_seps, {
                        'data': 'Val',
                        'loss': loss_valid,
                        'metric': metric_valid,
                        'logmetrics': log_metrics_valid
                    })

            # Stopped by overfit detector
            if self.stopper.stop():
                if verbose:
                    print("[{}] Training stopped by overfit detector. ({}/{})".format(
                        self.serial, self.current_epoch-self.stopper.state()[1]+1, self.max_epochs))
                    print(
                        f"[{self.serial}] Best score is {self.stopper.score():.{self.round_float}f}")
                load_snapshots_to_model(str(snapshot_path), self.model, self.optimizer)

                if calibrate_model:
                    if loader_valid is None:
                        print('loader_valid is necessary for calibration.')
                    else:
                        self.calibrate_model(loader_valid)

                if predict_valid:
                    self.oof = self.predict(
                        loader_valid, test_time_augmentations=test_time_augmentations, verbose=verbose)
                if predict_test:
                    self.pred = self.predict(
                        loader_test, test_time_augmentations=test_time_augmentations, verbose=verbose)
                break

            # TODO: Callback a posteriori

        else:  # Not stopped by overfit detector
            if verbose:
                print(
                    f"[{self.serial}] Best score is {self.stopper.score():.{self.round_float}f}")
            load_snapshots_to_model(str(snapshot_path), self.model, self.optimizer)

            if calibrate_model:
                if loader_valid is None:
                    print('loader_valid is necessary for calibration.')
                else:
                    self.calibrate_model(loader_valid)

            if predict_valid:
                if loader_valid is None:
                    self.oof = self.predict(
                        loader, test_time_augmentations=test_time_augmentations, verbose=verbose)
                else:
                    self.oof = self.predict(
                        loader_valid, test_time_augmentations=test_time_augmentations, verbose=verbose)
            if predict_test:
                self.pred = self.predict(
                    loader_test, test_time_augmentations=test_time_augmentations, verbose=verbose)
    
    fit = train # for compatibility

    def calibrate_model(self, loader):
        self.model = TemperatureScaler(self.model).to(self.device)
        self.model.set_temperature(loader)
