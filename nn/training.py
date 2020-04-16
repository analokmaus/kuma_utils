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

try:
    from torchsummary import summary
except:
    print('torch summary not found.')


'''
Misc.
'''

def scan_requires_grad(model):
    frozen = 0
    unfrozen = 0
    for i, param in enumerate(model.parameters()):
        if param.requires_grad:
            unfrozen += 1
        else:
            frozen += 1
    return frozen, unfrozen


def set_requires_grad(model, requires_grad=True, verbose=True):
    for i, param in enumerate(model.parameters()):
        param.requires_grad = requires_grad
    if verbose:
        frozen, unfrozen = scan_requires_grad(model)
        print(f'{frozen}/{frozen+unfrozen} params is frozen.')
        

'''
Stopper

# Methods
__call__() : bool     = return whether score is updated
stop() : bool         = return whether to stop training or not
state() : int, int    = return current / total
score() : float       = return best score
freeze()              = update score but never stop
unfreeze()            = unset freeze()
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


class EarlyStopping:
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
        self.val_loss_min = np.Inf
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


'''
Event
'''

class DummyEvent:
    ''' Dummy event does nothing '''

    def __init__(self):
        pass

    def __call__(self, model, optimizer, scheduler, stopper, 
                 criterion, eval_metric, i_epoch, log):
        return model, optimizer, scheduler, stopper, criterion, eval_metric


'''
Trainer
'''

class NeuralTrainer:  # depreciated
    '''
    # Important:
    This one is depreciated. Use TorchTrainer instead.

    Trainer for pytorch models
    '''

    def __init__(self, model, optimizer, scheduler, device=None, tta=1):
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        self.model = model
        self.device = device
        self.model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, 
              loader, criterion, num_epochs, snapshot_path, 
              loader_valid=None, loader_test=None, tta=1,
              eval_metric=None, 
              logger=None, event=None, stopper=None,
              resume=False,
              grad_accumulations=1, eval_interval=1, 
              log_interval=10, log_verbosity=1, 
              predict_oof=True, predict_pred=True,
              name='', verbose=True):

        if logger is None:
            logger = DummyLogger('')
        if event is None:
            event = DummyEvent()
        if stopper is None:
            stopper = DummyStopper()
        if eval_metric is None:
            eval_metric = lambda x, y: -1 * criterion(x, y).item()
            if verbose:
                print(f'[NT] eval_metric is not set. inversed criterion will be used instead.')

        start_epoch = 0
        self.tta = tta # test time augmentation
        self.log = defaultdict(dict)
        self.log['train']['loss'] = []
        self.log['train']['score'] = []
        self.log['valid']['loss'] = []
        self.log['valid']['score'] = []

        if resume:
            load_snapshots_to_model(snapshot_path, 
                self.model, self.optimizer, self.scheduler)
            start_epoch = load_epoch(snapshot_path)
            if verbose:
                print(f'[NT] {snapshot_path} is loaded. Continuing from epoch {start_epoch}.')

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
            if verbose:
                print(f'[NT] {torch.cuda.device_count()} gpus found.')

        _align = len(str(start_epoch+num_epochs))
        
        for _epoch, epoch in enumerate(range(start_epoch, start_epoch+num_epochs)):
            start_time = time.time()
            self.scheduler.step()

            '''
            Run event
            '''
            self.model, self.optimizer, self.scheduler, stopper, criterion, eval_metric = \
                event(self.model, self.optimizer, self.scheduler, 
                      stopper, criterion, eval_metric, _epoch, self.log)
            
            '''
            Train
            '''
            loss_train = 0.0
            score_train = 0.0
            total_batch = len(loader.dataset) / loader.batch_size

            self.model.train()
            for batch_i, (X, y) in enumerate(loader):
                batches_done = len(loader) * epoch + batch_i

                X = X.to(self.device)
                _y = self.model(X)
                y = y.to(self.device)
                loss = criterion(_y, y)
                loss.backward()

                if _epoch == 0 and batch_i == 0:
                    # Save output dimension in the first run
                    self.out_dim = _y.shape[1:]

                if batches_done % grad_accumulations == 0:
                    # Accumulates gradient before each step
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if batch_i % log_interval == 0:
                    score = eval_metric(_y, y)
                    for param_group in self.optimizer.param_groups:
                        learning_rate = param_group['lr']
                    evaluation_metrics = [
                        (f'score_{name}', score),
                        (f'loss_{name}', loss.item()),
                        (f'lr_{name}', learning_rate)
                    ]
                    logger.list_of_scalars_summary(evaluation_metrics, batches_done)

                batch_weight = len(X) / loader.batch_size
                loss_train += loss.item() / total_batch * batch_weight
                score_train += score / total_batch * batch_weight

            self.log['train']['loss'].append(loss_train)
            self.log['train']['score'].append(score_train)

            current_time = time.strftime(
                '%H:%M:%S', time.gmtime()) + ' ' if verbose >= 20 else ''
            log_str = f'{current_time}[{epoch+1:0{_align}d}/{start_epoch+num_epochs}] Trn '
            log_str += f"loss={loss_train:.6f} score={score_train:.6f}"

            '''
            No validation set
            '''
            if loader_valid is None:
                early_stopping_target = score_train

                if stopper(early_stopping_target): # score improved
                    save_snapshots(epoch, 
                        self.model, self.optimizer, self.scheduler, snapshot_path)
                    log_str += f' best={stopper.score():.6f}'
                else:
                    log_str += f' best={stopper.score():.6f}*({stopper.state()[0]})'

                if verbose and epoch % log_verbosity == 0:
                    print(log_str)

                if stopper.stop():
                    print("[NT] Training stopped by overfit detector. ({}/{})".format(
                        epoch-stopper.state()[1]+1, start_epoch+num_epochs))
                    print(f"[NT] Best score is {stopper.score():.6f}")
                    load_snapshots_to_model(str(snapshot_path), self.model)
                    if predict_oof:
                        self.oof = self.predict(loader, verbose=verbose)
                    if predict_pred:
                        self.pred = self.predict(loader_test, verbose=verbose)
                    break

                continue

            '''
            Validation
            '''
            if verbose and int(str(verbose)[-1]) >= 2 and epoch % log_verbosity == 0:  
                # Show log of training set
                print(log_str)

            if epoch % eval_interval == 0:
                loss_valid = 0.0
                score_valid = 0.0
                total_batch = len(loader_valid.dataset) / loader_valid.batch_size
                self.model.eval()
                with torch.no_grad():
                     for X, y in loader_valid:
                        X = X.to(self.device)
                        _y = self.model(X)
                        y = y.to(self.device)
                        loss = criterion(_y, y)
                        score = eval_metric(_y, y)

                        batch_weight = len(X) / loader_valid.batch_size
                        loss_valid += loss.item() / total_batch * batch_weight
                        score_valid += score / total_batch * batch_weight

                self.log['valid']['loss'].append(loss_valid)
                self.log['valid']['score'].append(score_valid)
                
                current_time = time.strftime(
                    '%H:%M:%S', time.gmtime()) + ' ' if verbose >= 20 else ''
                log_str = f'{current_time}[{epoch+1:0{_align}d}/{start_epoch+num_epochs}] Val '
                log_str += f"loss={loss_valid:.6f} score={score_valid:.6f}"
                evaluation_metrics = [
                    (f"score_valid({name})", score_valid),
                    (f"loss_valid({name})", loss_valid)
                ]
                logger.list_of_scalars_summary(evaluation_metrics, epoch)

                early_stopping_target = score_valid
                if stopper(early_stopping_target):  # score updated
                    save_snapshots(
                        epoch, self.model, self.optimizer, self.scheduler, snapshot_path)
                    log_str += f' best={stopper.score():.6f}'
                else:
                    log_str += f' best={stopper.score():.6f}*({stopper.state()[0]})'

                if verbose and epoch % log_verbosity == 0:
                    print(log_str)

            if stopper.stop():
                print("[NT] Training stopped by overfit detector. ({}/{})".format(
                    epoch-stopper.state()[1]+1, start_epoch+num_epochs))
                print(f"[NT] Best score is {stopper.score():.6f}")
                load_snapshots_to_model(str(snapshot_path), self.model)
                if predict_oof:
                    self.oof = self.predict(loader_valid, verbose=verbose)
                if predict_pred:
                    self.pred = self.predict(loader_test, verbose=verbose)
                break

        else: # Not stopped by overfit detector
            print(f"[NT] Best score is {stopper.score():.6f}")
            load_snapshots_to_model(str(snapshot_path), self.model)
            if predict_oof:
                if loader_valid is None:
                    self.oof = self.predict(loader, verbose=verbose)
                else:
                    self.oof = self.predict(loader_valid, verbose=verbose)
            if predict_pred:
                self.pred = self.predict(loader_test, verbose=verbose)

    def predict(self, loader, path=None, verbose=True):
        if loader is None:
            print('[NT] No data loaded. Skipping prediction...')
            return None
        if self.tta < 1:
            self.tta = 1
        batch_size = loader.batch_size
        prediction = np.zeros(
            (len(loader.dataset), *self.out_dim), dtype=np.float16)

        self.model.eval()
        with torch.no_grad():
            for _epoch in range(self.tta):
                for batch_i, (X, _) in enumerate(loader):
                    X = X.to(self.device)
                    _y = self.model(X).detach()
                    _y = _y.cpu().numpy()
                    idx = (batch_i*batch_size, (batch_i+1)*batch_size)
                    prediction[idx[0]:idx[1]] = _y / self.tta
        
        if path is not None:
            np.save(path, prediction)

        if verbose:
            print(f'[NT] Prediction done. exported to {path}')

        return prediction

    def info(self, input_shape):
        try:
            print(summary(self.model, input_shape))
        except:
            print('')


class TorchTrainer:
    '''
    Simple Trainer for PyTorch models

    # Usage
    model = Net()
    NN_FIT_PARAMS = {
        'loader': loader_train,
        'loader_valid': loader_valid,
        'loader_test': loader_test,
        'criterion': nn.BCEWithLogitsLoss(),
        'optimizer': optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-3),
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
                 model, device=None, serial='Trainer'):
        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = model
        self.device = device
        self.serial = serial
        self.model.to(self.device)
        print(f'[{self.serial}] Model is on {self.device}.')

    def train_loop(self, loader, grad_accumulations=1, logger_interval=1):
        loss_total = 0.0
        metric_total = 0.0
        total_batch = len(loader.dataset) / loader.batch_size

        self.model.train()
        for batch_i, (X, y) in enumerate(loader):
            batches_done = len(loader) * self.current_epoch + batch_i

            X = X.to(self.device)
            _y = self.model(X)
            y = y.to(self.device)
            loss = self.criterion(_y, y)
            loss.backward()

            if batch_i == 0:
                # Save output dimension in the first run
                self.out_dim = _y.shape[1:]

            if batches_done % grad_accumulations == 0:
                # Accumulates gradient before each step
                self.optimizer.step()
                self.optimizer.zero_grad()

            if batch_i % logger_interval == 0:
                metric = self.eval_metric(_y, y)
                for param_group in self.optimizer.param_groups:
                    learning_rate = param_group['lr']
                log_train_batch = [
                    (f'batch_metric_train[{self.serial}]', metric),
                    (f'batch_loss_train[{self.serial}]', loss.item()),
                    (f'batch_lr_train[{self.serial}]', learning_rate)
                ]
                self.logger.list_of_scalars_summary(log_train_batch, batches_done)

            batch_weight = len(X) / loader.batch_size
            loss_total += loss.item() / total_batch * batch_weight
            metric_total += metric / total_batch * batch_weight
        
        log_train = [
            (f'epoch_metric_train[{self.serial}]', metric_total),
            (f'epoch_loss_train[{self.serial}]', loss_total)
        ]
        self.logger.list_of_scalars_summary(log_train, self.current_epoch)
        self.log['train']['loss'].append(loss_total)
        self.log['train']['metric'].append(metric_total)
        
        return loss_total, metric_total

    def valid_loop(self, loader, grad_accumulations=1, logger_interval=1):
        loss_total = 0.0
        metric_total = 0.0
        total_batch = len(loader.dataset) / loader.batch_size

        self.model.eval()
        with torch.no_grad():
            for X, y in loader:
                X = X.to(self.device)
                _y = self.model(X)
                y = y.to(self.device)
                loss = self.criterion(_y, y)
                metric= self.eval_metric(_y, y)

                batch_weight = len(X) / loader.batch_size
                loss_total += loss.item() / total_batch * batch_weight
                metric_total += metric / total_batch * batch_weight

        log_valid = [
            (f'epoch_metric_valid[{self.serial}]', metric_total),
            (f'epoch_loss_valid[{self.serial}]', loss_total)
        ]
        self.logger.list_of_scalars_summary(log_valid, self.current_epoch)
        self.log['valid']['loss'].append(loss_total)
        self.log['valid']['metric'].append(metric_total)

        return loss_total, metric_total

    def predict(self, loader, path=None, test_time_augmentations=1, verbose=True):
        if loader is None:
            print(f'[{self.serial}] No data to predict. Skipping prediction...')
            return None
        batch_size = loader.batch_size
        prediction = np.zeros(
            (len(loader.dataset), *self.out_dim), dtype=np.float16)

        self.model.eval()
        with torch.no_grad():
            for _epoch in range(test_time_augmentations):
                for batch_i, (X, _) in enumerate(loader):
                    X = X.to(self.device)
                    _y = self.model(X).detach()
                    _y = _y.cpu().numpy()
                    idx = slice(batch_i*batch_size, (batch_i+1)*batch_size)
                    prediction[idx] = _y / test_time_augmentations

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
                log_str += f'{item}={info[item]:.6f}'
            elif item == 'epoch':
                align = len(str(self.max_epochs))
                log_str += f'E{self.current_epoch:0{align}d}/{self.max_epochs}'
            elif item == 'earlystopping':
                counter, patience = self.stopper.state()
                best = self.stopper.score()
                if best is not None:
                    log_str += f'best={best:.6f}'
                    if counter > 0:
                        log_str += f'*({counter}/{patience})'
            log_str += sep
        if len(log_str) > 0:
            print(log_str)

    def fit(self,
            # Essential
            criterion, optimizer, scheduler, 
            loader, num_epochs, loader_valid=None, loader_test=None,
            snapshot_path=None, resume=False,  # Snapshot
            multi_gpu=True, grad_accumulations=1,  # Train
            eval_metric=None, eval_interval=1,  # Evaluation
            test_time_augmentations=1, predict_valid=True, predict_test=True,  # Prediction
            event=DummyEvent(), stopper=DummyStopper(),  # Train add-ons
            # Logger and info
            logger=DummyLogger(''), logger_interval=1, 
            info_train=True, info_valid=True, info_interval=1, 
            info_format='epoch time data loss metric earlystopping', verbose=True):

        if eval_metric is None:
            eval_metric = lambda x, y: -1 * criterion(x, y).item()
            print(f'[{self.serial}] eval_metric is not set. Inversed criterion will be used instead.')

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.eval_metric = eval_metric
        self.logger = logger
        self.event = event
        self.stopper = stopper

        self.current_epoch = 0
        self.log = {
            'train': {'loss': [], 'metric': []},
            'valid': {'loss': [], 'metric': []}
        }
        info_items = re.split(r'[^a-z]+', info_format)
        info_seps = re.split(r'[a-z]+', info_format)
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

        if resume:
            load_snapshots_to_model(snapshot_path, self.model, self.optimizer, self.scheduler)
            self.current_epoch = load_epoch(snapshot_path)
            if verbose:
                print(
                    f'[{self.serial}] {snapshot_path} is loaded. Continuing from epoch {start_epoch}.')

        if multi_gpu and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
            if verbose:
                print(f'[{self.serial}] {torch.cuda.device_count()} gpus found.')

        self.max_epochs = self.current_epoch + num_epochs

        for epoch in range(num_epochs):
            self.current_epoch += 1
            start_time = time.time()
            self.scheduler.step()

            ### Event
            self.model, self.optimizer, self.scheduler, self.stopper, self.criterion, self.eval_metric = \
                event(self.model, self.optimizer, self.scheduler,
                      self.stopper, self.criterion, self.eval_metric, epoch, self.log)

            ### Training
            loss_train, metric_train = self.train_loop(loader, grad_accumulations, logger_interval)

            ### No validation set
            if loader_valid is None:
                early_stopping_target = metric_train

                if self.stopper(early_stopping_target):  # score improved
                    save_snapshots(self.current_epoch,
                                   self.model, self.optimizer, self.scheduler, snapshot_path)
                
                if info_train and epoch % info_interval == 0:
                    self.print_info(info_items, info_seps, {
                        'data': 'Trn',
                        'loss': loss_train,
                        'metric': metric_train,
                    })

                if self.stopper.stop():
                    if verbose:
                        print("[{}] Training stopped by overfit detector. ({}/{})".format(
                            self.serial, self.current_epoch-self.stopper.state()[1]+1, self.max_epochs))
                        print(f"[{self.serial}] Best score is {self.stopper.score():.6f}")
                    load_snapshots_to_model(str(snapshot_path), self.model)
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
                })

            ### Validation
            if epoch % eval_interval == 0:
                loss_valid, metric_valid = self.valid_loop(loader_valid, grad_accumulations, logger_interval)
                
                early_stopping_target = metric_valid
                if self.stopper(early_stopping_target):  # score improved
                    save_snapshots(
                        self.current_epoch, self.model, self.optimizer, self.scheduler, snapshot_path)

                if info_valid and epoch % info_interval == 0:
                    self.print_info(info_items, info_seps, {
                        'data': 'Val',
                        'loss': loss_valid,
                        'metric': metric_valid,
                    })

            # Stooped by overfit detector
            if self.stopper.stop():
                if verbose:
                    print("[{}] Training stopped by overfit detector. ({}/{})".format(
                        self.serial, self.current_epoch-self.stopper.state()[1]+1, self.max_epochs))
                    print(f"[{self.serial}] Best score is {self.stopper.score():.6f}")
                load_snapshots_to_model(str(snapshot_path), self.model)
                if predict_valid:
                    self.oof = self.predict(
                        loader_valid, test_time_augmentations=test_time_augmentations, verbose=verbose)
                if predict_test:
                    self.pred = self.predict(
                        loader_test, test_time_augmentations=test_time_augmentations, verbose=verbose)
                break

        else:  # Not stopped by overfit detector
            if verbose:
                print(f"[{self.serial}] Best score is {self.stopper.score():.6f}")
            load_snapshots_to_model(str(snapshot_path), self.model)
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


'''
Cross Validation for Tabular data
'''

class TorchCV: 
    IGNORE_PARAMS = [
        'loader', 'loader_valid', 'loader_test', 'snapshot_path', 'logger'
    ]

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
            group=None, n_splits=None, transform=None,
            eval_metric=None, batch_size=64, 
            snapshot_dir=None, logger=DummyLogger(''), 
            fit_params={}, verbose=True):
        
        if not isinstance(eval_metric, (list, tuple, set)):
            eval_metric = [eval_metric]
        if snapshot_dir is None:
            snapshot_dir = Path().cwd()
        if not isinstance(snapshot_dir, Path):
            snapshot_dir = Path(snapshot_dir)
        assert snapshot_dir.is_dir()
            
        if n_splits is None:
            K = self.datasplit.get_n_splits()
        else:
            K = n_splits

        self.oof = np.zeros(len(X), dtype=np.float)
        if X_test is not None:
            self.pred = np.zeros(len(X_test), dtype=np.float)
        self.imps = np.zeros((X.shape[1], K))
        self.scores = np.zeros((len(eval_metric), K))

        default_path = snapshot_dir/'default.pt'
        save_snapshots(0, self.model, fit_params['optimizer'], fit_params['scheduler'], default_path)
        template = {}
        for item in ['stopper', 'event']:
            if item in fit_params.keys():
                template[item] = deepcopy(fit_params[item])
        for item in self.IGNORE_PARAMS:
            if item in fit_params.keys():
                fit_params.pop(item)

        for fold_i, (train_idx, valid_idx) in enumerate(
            self.datasplit.split(X, y, group)):

            x_train, x_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]
            if X_test is not None:
                x_test = X_test.copy()

            if transform is not None:
                x_train, x_valid, y_train, y_valid, x_test = transform(
                    Xs=(x_train, x_valid), ys=(y_train, y_valid),
                    X_test=x_test)

            ds_train = numpy2dataset(x_train, y_train)
            ds_valid = numpy2dataset(x_valid, y_valid)
            ds_test = numpy2dataset(x_test, np.arange(len(x_test)))

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

            self.oof[valid_idx] = trainer_fold.oof.squeeze()
            if X_test is not None:
                self.pred += trainer_fold.pred.squeeze() / K

            for i, _metric in enumerate(eval_metric):
                score = _metric(y_valid, self.oof[valid_idx])
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
