import os
import re
import sys
import time
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from copy import copy, deepcopy

import numpy as np
import pandas as pd

import torch
import torch.utils.data as D
from .datasets import *
from .snapshot import *
from .logger import *
from .temperature_scaling import *
from .fp16util import network_to_half

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
Stopper

# Methods
__call__(score) : bool  = return whether score is improved
stop() : bool           = return whether to stop training or not
state() : int, int      = return current / total
score() : float         = return best score
freeze()                = update score but never stop
unfreeze()              = unset freeze()
'''

class DummyStopper:
    ''' No stopper '''

    def __init__(self):
        pass

    def __call__(self, val_loss):
        return True

    def stop(self):
        return False

    def state(self):
        return 0, 0

    def score(self):
        return 0.0

    def dump_state_dict(self):
        return {}

    def load_state_dict(self, checkpoint):
        pass

    def __repr__(self):
        return 'No Stopper'


class EarlyStopping(DummyStopper):
    '''
    Early stops the training if validation loss doesn't improve after a given patience.
    patience: int   = early stopping rounds
    maximize: bool  = whether maximize or minimize metric
    '''

    def __init__(self, patience=5, maximize=False):
        self.patience = patience
        self.counter = 0
        self.log = []
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        if maximize:
            self.coef = 1
        else:
            self.coef = -1
        self.frozen = False

    def __call__(self, val_loss):
        score = self.coef * val_loss
        self.log.append(score)
        if self.best_score is None:
            self.best_score = score
            return True
        elif score < self.best_score:
            if not self.frozen:
                self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else: # score improved
            self.best_score = score
            self.counter = 0
            return True

    def stop(self):
        return self.early_stop

    def state(self):
        return self.counter, self.patience
        
    def score(self):
        return self.best_score

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    def reset(self):
        self.best_score = None

    def dump_state_dict(self):
        return {
            'best_score': self.best_score,
            'counter': self.counter,
        }

    def load_state_dict(self, checkpoint):
        self.best_score = checkpoint['best_score']
        self.counter = checkpoint['counter']

    def __repr__(self):
        return f'EarlyStopping({self.patience})'


'''
Event
'''

class DummyEvent:
    ''' Dummy event does nothing '''

    def __init__(self):
        pass

    def __call__(self, **kwargs):
        pass

    def dump_state_dict(self):
        return {}

    def load_state_dict(self, checkpoint):
        pass

    def __repr__(self):
        return 'No Event'


class NoEarlyStoppingNEpochs(DummyEvent):

    def __init__(self, n):
        self.n = n

    def __call__(self, **kwargs):
        if kwargs['global_epoch'] == 0:
            kwargs['stopper'].freeze()
            kwargs['stopper'].reset()
            print(f"Epoch\t{kwargs['epoch']}: Earlystopping is frozen.")
        elif kwargs['global_epoch'] < self.n:
             kwargs['stopper'].reset()
        elif kwargs['global_epoch'] == self.n:
            kwargs['stopper'].unfreeze()
            print(f"Epoch\t{kwargs['epoch']}: Earlystopping is unfrozen.")

    def __repr__(self):
        return f'NoEarlyStoppingNEpochs({self.n})'


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
        
        self.device = device
        self.serial = serial
        self.is_fp16 = fp16 # Automatically use apex if available
        self.is_xla = xla
        self.apex_opt_level = 'O1'
        self.model = model
        self.all_inputs_to_model = False
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
                f'[{self.serial}] Multi parallel training for xla devices is WIP.')

        if torch.cuda.device_count() > 1:
            all_devices = list(range(torch.cuda.device_count()))
            if self.is_fp16 and APEX_FLAG:
                self.model = nn.parallel.DataParallel(self.model)
            else:
                self.model = nn.parallel.DataParallel(self.model)

            print(f'[{self.serial}] {torch.cuda.device_count()}({all_devices}) gpus found.')

    def train_loop(self, loader, grad_accumulations=1, logger_interval=1):
        loss_total = 0.0
        # metric_total = 0.0
        total_batch = len(loader.dataset) / loader.batch_size
        approx = []
        target = []

        self.model.train()
        for batch_i, inputs in enumerate(loader):
            batches_done = len(loader) * self.current_epoch + batch_i

            X, y = inputs[0], inputs[1]
            X = X.to(self.device)
            y = y.to(self.device)
            if len(inputs) == 3:
                z = inputs[2]
                z = z.to(self.device)
            
            if self.all_inputs_to_model:
                if len(inputs) == 3:
                    _y = self.model(X, y, z)
                elif len(inputs) == 2:
                    _y = self.model(X, y)
            else:
                _y = self.model(X)
            if self.is_fp16:
                _y = _y.float()
            approx.append(_y.clone().detach())
            target.append(y.clone().detach())
            
            if len(inputs) == 3:
                loss = self.criterion(_y, y, z)
            elif len(inputs) == 2:
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
                # metric = self.eval_metric(_y, y)
                for param_group in self.optimizer.param_groups:
                    learning_rate = param_group['lr']
                log_train_batch = [
                    # (f'batch_metric_train[{self.serial}]', metric),
                    (f'batch_loss_train[{self.serial}]', loss.item()),
                    (f'batch_lr_train[{self.serial}]', learning_rate)
                ]
                self.logger.list_of_scalars_summary(log_train_batch, batches_done)

            batch_weight = len(X) / loader.batch_size
            loss_total += loss.item() / total_batch * batch_weight
            # metric_total += metric / total_batch * batch_weight
        
        approx = torch.cat(approx).cpu()
        target = torch.cat(target).cpu()
        if self.eval_metric is None:
            metric_total = -loss_total
        else:
            metric_total = self.eval_metric(approx, target)
        log_metrics_total = []
        for log_metric in self.log_metrics:
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

        self.model.eval()
        with torch.no_grad():
            for inputs in loader:
                X, y = inputs[0], inputs[1]
                X = X.to(self.device)
                y = y.to(self.device)
                if len(inputs) == 3:
                    z = inputs[2]
                    z = z.to(self.device)
              
                if self.all_inputs_to_model:
                    if len(inputs) == 3:
                        _y = self.model(X, y, z)
                    elif len(inputs) == 2:
                        _y = self.model(X, y)
                else:
                    _y = self.model(X)
                
                if self.is_fp16:
                    _y = _y.float()
                approx.append(_y.clone().detach())
                target.append(y.clone().detach())
            
                if len(inputs) == 3:
                    loss = self.criterion(_y, y, z)
                elif len(inputs) == 2:
                    loss = self.criterion(_y, y)

                batch_weight = len(X) / loader.batch_size
                loss_total += loss.item() / total_batch * batch_weight

        approx = torch.cat(approx).cpu()
        target = torch.cat(target).cpu()
        if self.eval_metric is None:
            metric_total = -loss_total
        else:
            metric_total = self.eval_metric(approx, target)
        log_metrics_total = []
        for log_metric in self.log_metrics:
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
            for tta_fold in range(test_time_augmentations):
                fold_prediction = []
                for batch_i, inputs in enumerate(loader):
                    X = inputs[0]
                    X = X.to(self.device)
                    if self.is_fp16 and APEX_FLAG:
                        with amp.disable_casts():
                            _y = self.model(X)
                    else:
                        _y = self.model(X)
                    fold_prediction.append(_y.detach())
                fold_prediction = torch.cat(fold_prediction).cpu().numpy()
                prediction.append(fold_prediction)

        if test_time_augmentations > 1:
            prediction = np.stack(prediction).mean(0)
        else:
            prediction = prediction[0]

        if path is not None:
            np.save(path, prediction)

        if verbose:
            print(f'[{self.serial}] Prediction done. (tta = {test_time_augmentations})')

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

    def fit(self,
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
            info_format='epoch time data loss metric logmetrics earlystopping', verbose=True):

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

            ### Event
            event(**{'model': self.model, 'optimizer': self.optimizer, 'scheduler': self.scheduler,
                     'stopper': self.stopper, 'criterion': self.criterion, 'eval_metric': self.eval_metric, 
                     'epoch': epoch, 'global_epoch': self.current_epoch, 'log': self.log})

            ### Training
            loss_train, metric_train, log_metrics_train = self.train_loop(loader, grad_accumulations, logger_interval)

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
                            loader, test_time_augmentations=1, verbose=verbose)
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
                loss_valid, metric_valid, log_metrics_valid = self.valid_loop(loader_valid, grad_accumulations, logger_interval)
                
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
                        loader_valid, test_time_augmentations=1, verbose=verbose)
                if predict_test:
                    self.pred = self.predict(
                        loader_test, test_time_augmentations=test_time_augmentations, verbose=verbose)
                break

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
                        loader, test_time_augmentations=1, verbose=verbose)
                else:
                    self.oof = self.predict(
                        loader_valid, test_time_augmentations=1, verbose=verbose)
            if predict_test:
                self.pred = self.predict(
                    loader_test, test_time_augmentations=test_time_augmentations, verbose=verbose)
    
    def calibrate_model(self, loader):
        self.model = TemperatureScaler(self.model).to(self.device)
        self.model.set_temperature(loader)


'''
Cross Validation for Tabular data
'''

class TorchCV: 
    IGNORE_PARAMS = {
        'loader', 'loader_valid', 'loader_test', 'snapshot_path', 'logger'
    }
    TASKS = {'binary', 'regression'}

    def __init__(self, model, datasplit, device=None):
        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model
        self.device = device
        self.datasplit = datasplit

        self.models = []
        self.oof = None
        self.pred = None
        self.imps = None

    def run(self, X, y, X_test=None,
            group=None, transform=None, task='binary', 
            eval_metric=None, batch_size=64, n_splits=None,
            snapshot_dir=None, logger=DummyLogger(''), 
            fit_params={}, verbose=True):
        
        if not isinstance(eval_metric, (list, tuple, set)):
            eval_metric = [eval_metric]
        if snapshot_dir is None:
            snapshot_dir = Path().cwd()
        if not isinstance(snapshot_dir, Path):
            snapshot_dir = Path(snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        assert snapshot_dir.is_dir()
        assert task in self.TASKS
            
        if n_splits is None:
            K = self.datasplit.get_n_splits()
        else:
            K = n_splits

        self.imps = np.zeros((X.shape[1], K))
        self.scores = np.zeros((len(eval_metric), K))
        self.numpy2dataset = Numpy2Dataset(task)

        default_path = snapshot_dir/'default.pt'
        save_snapshots(default_path, 0, self.model, fit_params['optimizer'], fit_params['scheduler'])
        template = {}
        for item in ['stopper', 'event']:
            if item in fit_params.keys():
                template[item] = deepcopy(fit_params[item])
        for item in self.IGNORE_PARAMS:
            if item in fit_params.keys():
                fit_params.pop(item)

        for fold_i, (train_idx, valid_idx) in enumerate(
            self.datasplit.split(X, y, group)):

            Xs = {'train': X[train_idx], 'valid': X[valid_idx],
                  'test': X_test.copy() if X_test is not None else None}
            ys = {'train': y[train_idx], 'valid': y[valid_idx]}

            if transform is not None:
                transform(Xs, ys)

            ds_train = self.numpy2dataset(Xs['train'], ys['train'])
            ds_valid = self.numpy2dataset(Xs['valid'], ys['valid'])
            ds_test = self.numpy2dataset(Xs['test'], np.arange(len(Xs['test'])))

            loader_train = D.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
            loader_valid = D.DataLoader(ds_valid, batch_size=batch_size, shuffle=False)
            loader_test = D.DataLoader(ds_test, batch_size=batch_size, shuffle=False)
            
            params_fold = copy(fit_params)
            params_fold.update({
                'loader': loader_train,
                'loader_valid': loader_valid,
                'loader_test': loader_test,
                'snapshot_path': snapshot_dir / f'snapshot_fold_{fold_i}.pt',
                'logger': logger
            })
            params_fold.update(deepcopy(template))
            load_snapshots_to_model(default_path, self.model, params_fold['optimizer'], params_fold['scheduler'])

            trainer_fold = TorchTrainer(self.model, self.device, serial=f'fold_{fold_i}')
            trainer_fold.fit(**params_fold)

            if fold_i == 0: # Initialize oof and prediction
                self.oof = np.zeros((len(X), trainer_fold.oof.shape[1]), dtype=np.float)
                if X_test is not None:
                    self.pred = np.zeros((len(X_test), trainer_fold.pred.shape[1]), dtype=np.float)

            self.oof[valid_idx] = trainer_fold.oof
            if X_test is not None:
                self.pred += trainer_fold.pred / K

            for i, _metric in enumerate(eval_metric):
                score = _metric(ys['valid'], self.oof[valid_idx])
                self.scores[i, fold_i] = score

            if verbose >= 0:
                log_str = f'[CV] Fold {fold_i+1}:'
                log_str += ''.join(
                    [f' m{i}={self.scores[i, fold_i]:.5f}' for i in range(len(eval_metric))])
                print(log_str)
        
        log_str = f'[CV] Overall:'
        log_str += ''.join(
            [f' m{i}={me:.5f}±{se:.5f}' for i, (me, se) in enumerate(zip(
                np.mean(self.scores, axis=1),
                np.std(self.scores, axis=1)/np.sqrt(len(eval_metric))
            ))]
        )
        print(log_str)
