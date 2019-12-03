import os
import sys
import time
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
from .snapshot import *
from .logger import *

try:
    from torchsummary import summary
except:
    print('torch summary not found.')


'''
Automation
'''

def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


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

    def __call__(self, val_loss):

        score = self.coef * val_loss

        if self.best_score is None:
            self.best_score = score
            return True
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.counter = 0
            return True


class NeuralTrainer:

    def __init__(self, model, optimizer, scheduler, 
                 stopper=None):
        self.model = model
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.earlystop = stopper
        
        # General params
        self.tta = 1

    def train(self, 
              loader, criterion, epochs, snapshot_path, 
              loader_valid=None, loader_test=None,
              logger=None, verbose=True, 
              eval_metric=None, resume=False,
              grad_accumulations=1, eval_interval=1, 
              log_interval=10, name=''):

        if logger is None:
            logger = DummyLogger('')

        if eval_metric is None:
            eval_metric = lambda x, y: criterion(x, y).item()
            if verbose:
                print(f'[NT] eval_metric is not set. criterion will be used instead.')

        start_epoch = 0
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

        _align = len(str(start_epoch+epochs))

        for epoch in range(start_epoch, start_epoch + epochs):
            start_time = time.time()
            self.scheduler.step()
            

            '''
            Train
            '''
            train_losses = []
            train_scores = []
            self.model.train()

            for batch_i, (X, y) in enumerate(loader):
                batches_done = len(loader) * epochs + batch_i

                X = X.to(self.device)
                _y = self.model(X)
                y = y.to(self.device)
                loss = criterion(_y, y)
                loss.backward()

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

            if verbose >= 2:
                current_time = time.strftime(
                    '%H:%M:%S', time.gmtime()) + ' ' if verbose >= 3 else ''
                log_str = f'{current_time}[{epoch+1:0{_align}d}/{start_epoch + epochs}] T\t'
                log_str += f"loss={avg_loss_train:.6f}\t score={avg_score_train:.6f}"
                print(log_str)


            '''
            No validation set
            '''
            if loader_valid is None:
                early_stopping_target = avg_score_train

                if self.earlystop(early_stopping_target): 
                    save_snapshots(epoch, 
                        self.model, self.optimizer, self.scheduler, snapshot_path)

                if self.earlystop.early_stop:
                    print("[NT] Training stopped by overfit detector. ({}/{})".format(
                        epoch-self.earlystop.patience+1, start_epoch+epochs))
                    load_snapshots_to_model(str(snapshot_path), self.model)
                    self.preds = self.predict(loader, verbose=verbose)
                    break

                continue


            '''
            Validation
            '''
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
                log_str = f'{current_time}[{epoch+1:0{_align}d}/{start_epoch + epochs}] V\t'
                log_str += f"loss={avg_loss_valid:.6f}\t score={avg_score_valid:.6f}"
                evaluation_metrics = [
                    (f"val_accuracy_{name}", avg_score_valid),
                    (f"valid_loss_{name}", avg_loss_valid)
                ]
                logger.list_of_scalars_summary(evaluation_metrics, epoch)

                early_stopping_target = avg_score_valid
                if self.earlystop(early_stopping_target):  # score updated
                    save_snapshots(
                        epoch, self.model, self.optimizer, self.scheduler, snapshot_path)
                else:
                    log_str += f'*({self.earlystop.counter})'

                if verbose:
                    print(log_str)

            if self.earlystop.early_stop:
                print("[NT] Training stopped by overfit detector. ({}/{})".format(
                    epoch-self.earlystop.patience+1, start_epoch+epochs))
                print(f"[NT] Best score is {self.earlystop.best_score:.6f}")
                load_snapshots_to_model(str(snapshot_path), self.model)
                self.pred = self.predict(loader_test, verbose=verbose)
                self.oof = self.predict(loader_valid, verbose=verbose)
                break

        else: # Not stopped by overfit detector
            print(f"[NT] Best score is {self.earlystop.best_score:.6f}")
            load_snapshots_to_model(str(snapshot_path), self.model)
            self.pred = self.predict(loader_test, verbose=verbose)
            self.oof = self.predict(loader_valid, verbose=verbose)

    def predict(self, loader, path=None, verbose=True):
        if self.tta < 1:
            self.tta = 1
        batch_size = loader.batch_size
        prediction = np.zeros(
            (len(loader.dataset), ), dtype=np.float16)

        self.model.eval()
        with torch.no_grad():
            for _epoch in range(self.tta):
                for batch_i, (X, _) in enumerate(loader):
                    X = X.to(self.device)
                    _y = self.model(X).detach()
                    _y = _y.cpu().numpy().squeeze()
                    idx = (batch_i*batch_size, (batch_i+1)*batch_size)
                    prediction[idx[0]:idx[1]] = _y / self.tta
        
        if path is not None:
            np.save(path, prediction)

        if verbose:
            print(f'[NT] Prediction done. exported to {path}')

        return prediction

    def info(self, input_shape):
        try:
            summary(self.model, input_shape)
        except:
            print('')
