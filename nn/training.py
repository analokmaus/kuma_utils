import os
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
- __call__() : bool     = return whether score is updated
- stop() : bool         = return whether to stop training or not
- state() : int, int    = return current / total
- score() : float       = return best score
- freeze()              = update score but never stop
- unfreeze()            = unset freeze()
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
    patience: int   = how long to wait after last time validation loss improved
    '''

    def __init__(self, patience=5, maximize=False):
        self.patience = patience
        self.counter = 0
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
        if self.best_score is None:
            self.best_score = score
            return True
        elif score < self.best_score:
            if not self.frozen:
                self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
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

class NeuralTrainer:
    '''
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
              log_interval=10, name='', 
              predict_oof=True, predict_pred=True,
              verbose=True):

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
            train_losses = []
            train_scores = []
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
                        learing_rate = param_group['lr']
                    evaluation_metrics = [
                        (f'score_{name}', score),
                        (f'loss_{name}', loss.item()),
                        (f'lr_{name}', learing_rate)
                    ]
                    logger.list_of_scalars_summary(evaluation_metrics, batches_done)

                    train_scores.append(score)
                    train_losses.append(loss.item())

            avg_loss_train = np.average(train_losses)
            avg_score_train = np.average(train_scores)
            self.log['train']['loss'].append(avg_loss_train)
            self.log['train']['score'].append(avg_score_train)

            current_time = time.strftime(
                '%H:%M:%S', time.gmtime()) + ' ' if verbose >= 3 else ''
            log_str = f'{current_time}[{epoch+1:0{_align}d}/{start_epoch+num_epochs}] Trn '
            log_str += f"loss={avg_loss_train:.6f} score={avg_score_train:.6f}"

            '''
            No validation set
            '''
            if loader_valid is None:
                early_stopping_target = avg_score_train

                if stopper(early_stopping_target): # score updated
                    save_snapshots(epoch, 
                        self.model, self.optimizer, self.scheduler, snapshot_path)
                    log_str += f' best={stopper.score():.6f}'
                else:
                    log_str += f' best={stopper.score():.6f}*({stopper.state()[0]})'

                if verbose >= 2:
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
            if verbose >= 2:  # Show log of training
                print(log_str)

            if epoch % eval_interval == 0:
                valid_losses = []
                valid_scores = []
                self.model.eval()
                with torch.no_grad():
                     for X, y in loader_valid:
                        X = X.to(self.device)
                        _y = self.model(X)
                        y = y.to(self.device)
                        loss = criterion(_y, y)
                        score = eval_metric(_y, y)

                        valid_scores.append(score)
                        valid_losses.append(loss.item())

                avg_loss_valid = np.average(valid_losses)
                avg_score_valid = np.average(valid_scores)
                self.log['valid']['loss'].append(avg_loss_valid)
                self.log['valid']['score'].append(avg_score_valid)
                
                current_time = time.strftime(
                    '%H:%M:%S', time.gmtime()) + ' ' if verbose >= 3 else ''
                log_str = f'{current_time}[{epoch+1:0{_align}d}/{start_epoch+num_epochs}] Val '
                log_str += f"loss={avg_loss_valid:.6f} score={avg_score_valid:.6f}"
                evaluation_metrics = [
                    (f"val_accuracy_{name}", avg_score_valid),
                    (f"valid_loss_{name}", avg_loss_valid)
                ]
                logger.list_of_scalars_summary(evaluation_metrics, epoch)

                early_stopping_target = avg_score_valid
                if stopper(early_stopping_target):  # score updated
                    save_snapshots(
                        epoch, self.model, self.optimizer, self.scheduler, snapshot_path)
                    f' best={stopper.score():.6f}'
                else:
                    log_str += f' best={stopper.score():.6f}*({stopper.state()[0]})'

                if verbose:
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


'''
Cross Validation for Tabular data
'''

class NeuralCV:

    def __init__(self, trainer, datasplit):
        self.trainer = trainer
        self.datasplit = datasplit
        self.models = []
        self.oof = None
        self.pred = None
        self.imps = None

    def run(self, X, y, X_test=None,
            group=None, n_splits=None,
            eval_metric=None, batch_size=None, 
            transform=None, train_params={}, verbose=True):
        
        if not isinstance(eval_metric, (list, tuple, set)):
            eval_metric = [eval_metric]
        
        if n_splits is None:
            K = self.datasplit.get_n_splits()
        else:
            K = n_splits

        self.oof = np.zeros(len(X), dtype=np.float)

        if X_test is not None:
            self.pred = np.zeros(len(X_test), dtype=np.float)

        self.imps = np.zeros((X.shape[1], K))
        self.scores = np.zeros((len(eval_metric), K))

        if batch_size is None:
            batch_size = 256

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

            nt = deepcopy(self.trainer)
            nt.train(loader=loader_train, 
                     loader_valid=loader_valid, 
                     loader_test=loader_test,
                     **deepcopy(train_params))

            self.oof[valid_idx] = nt.oof.squeeze()
            if X_test is not None:
                self.pred += nt.pred.squeeze() / K

            for i, _metric in enumerate(eval_metric):
                score = _metric(y_valid, self.oof[valid_idx])
                self.scores[i, fold_i] = score

            if verbose >= 0:
                log_str = f'[CV] Fold {fold_i}:'
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
