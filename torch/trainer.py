from pathlib import Path
from tqdm import tqdm
from copy import copy, deepcopy
from collections import defaultdict
import time
import pickle
import subprocess
import inspect
import sys

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.nn import SyncBatchNorm

from .utils import XLA, get_device, set_random_seeds
from .tb_logger import DummyTensorBoardLogger
from .temperature_scaling import TemperatureScaler
from .callbacks import TorchLogger, EarlyStopping
from .hooks import SimpleHook
from . import distributed as comm

try:
    from torch.cuda import amp
    AMP = True
except ModuleNotFoundError:
    AMP = False


DDP_TMP_PATH = Path('.ddp_torch_tmp')


class TorchTrainer:
    '''
    Simple Trainer for PyTorch models
    
    This is something similar to PyTorch Lightning, but this works with vanilla PyTorch modules.

    TODO:
    - DDP for xla device
    '''

    def __init__(self, 
                 model, device=None, serial='trainer0'):
        
        self.serial = serial
        self.device, self.device_ids = get_device(device)
        self.world_size = len(self.device_ids)
        self.xla = self.device.type == 'xla'
        self.model = model
        self.rank = 0

        ### Some implicit attributes
        # DDP
        self.ddp_sync_batch_norm = True
        self.ddp_average_loss = True
        self.ddp_workers = -1
        # MISC
        self.scheduler_target = None
        self.progress_bar = False
        self.debug = False

    def _register_callbacks(self, callbacks):
        self.before_train = [func.before_train for func in callbacks]
        self.after_train = [func.after_train for func in callbacks]

    def _register_hook(self, hook):
        self.forward_train = hook.forward_train
        self.forward_test = hook.forward_test
        self.evaluate_epoch = hook.evaluate_epoch

    def _configure_model(self):
        ''' Mixed precision '''
        if self.fp16:
            if AMP:
                if self.rank == 0:
                    self.logger('Mixed precision training on torch amp.')
            else:
                self.fp16 = False
                if self.rank == 0:
                    self.logger('No mixed precision training backend found.')

        ''' Parallel training '''
        if self.parallel == 'dp':
            if self.xla:
                raise NotImplementedError(
                    'Data Parallel on xla devices is not supported.')
            else:
                self.model = nn.parallel.DataParallel(
                    self.model, device_ids=self.device_ids).to(self.device)
                self.logger(f'DataParallel on devices {self.device_ids}')

        elif self.parallel == 'ddp':
            if self.xla:
                raise NotImplementedError(
                    'Distributed Data Parallel on xla devices is WIP.')
            else:
                if self.ddp_sync_batch_norm:
                    self.model = SyncBatchNorm.convert_sync_batchnorm(self.model)
                self.model = DistributedDataParallel(
                    self.model.to(self.rank), device_ids=[self.rank],
                    broadcast_buffers=False,
                    find_unused_parameters=True
                )
                if self.rank == 0:
                    self.logger(f'DistributedDataParallel on devices {self.device_ids}')

        elif self.parallel is not None:
            raise ValueError(f'Unknown type of parallel {self.parallel}')

        else: # Single 
            self.model.to(self.device)
            self.logger(f'Model on {self.device}')
 
    def _configure_loader_ddp(self, loader, shuffle=True):
        if loader is None:
            return None
        skip_keys = ['sampler', 'batch_sampler', 'dataset_kind']
        dl_args = {
            k: v for k, v in loader.__dict__.items() \
                if not k.startswith('_') and k not in skip_keys
        }
        sampler = DistributedSampler(
            loader.dataset, num_replicas=self.world_size, rank=self.rank, shuffle=shuffle)
        dl_args['sampler'] = sampler
        if self.ddp_workers == -1:
            dl_args['num_workers'] = int(dl_args['num_workers'] / self.world_size) 
        else:
            dl_args['num_workers'] = self.ddp_workers
        if dl_args['batch_size'] % self.world_size != 0:
            raise ValueError(f'batch size must be a multiple of world size({self.world_size}).')
        dl_args['batch_size'] = int(dl_args['batch_size'] / self.world_size)
        return type(loader)(**dl_args)

    def _train_one_epoch(self, loader):
        batch_weights = torch.ones(len(loader)).float().to(self.device)
        if len(loader.dataset) % loader.batch_size != 0:
            batch_weights[-1] = \
                (len(loader.dataset) % loader.batch_size)  / loader.batch_size
        #
        loader_time = .0
        train_time = .0
        curr_time = time.time()
        #
        # Storage
        self.epoch_storage = defaultdict(list)
        for key in ['approx', 'target', 'loss', 'batch_metric']:
            self.epoch_storage[key] = []
        if self.fp16:
            scaler = amp.GradScaler()

        self.model.train()
        if self.progress_bar and self.rank == 0:
            iterator = enumerate(tqdm(loader, desc='train'))
        else:
            iterator = enumerate(loader)

        for batch_i, inputs in iterator:
            loader_time += time.time() - curr_time
            curr_time = time.time()

            self.optimizer.zero_grad()
            batches_done = len(loader) * (self.global_epoch-1) + batch_i
            inputs = [t.to(self.device) for t in inputs]

            if self.fp16:
                with amp.autocast():
                    loss = self.forward_train(self, inputs)
                loss = loss / self.grad_accumulations
                scaler.scale(loss).backward()
                if (batch_i + 1) % self.grad_accumulations == 0:
                    scaler.step(self.optimizer)
                    scaler.update()
            else:
                loss = self.forward_train(self, inputs)
                loss = loss / self.grad_accumulations
                loss.backward()
                if (batch_i + 1) % self.grad_accumulations == 0:
                    if self.xla:
                        xm.optimizer_step(self.optimizer, barrier=True)
                    else:
                        self.optimizer.step()
                    if self.batch_scheduler:
                        self.scheduler.step()

            ''' TensorBoard logging '''
            if self.parallel == 'ddp' and self.ddp_average_loss:
                loss_batch = comm.gather_tensor(
                    loss.detach().clone().view(1)).mean().item()
            else: # Use loss on device: 0
                loss_batch = loss.item()
            learning_rate = [ param_group['lr'] \
                for param_group in self.optimizer.param_groups ]
            logs = [
                ('batch_train_loss', loss_batch),
                ('batch_train_lr', learning_rate)
            ]
            if len(self.epoch_storage['batch_metric']) > 0:
                metric = self.epoch_storage['batch_metric'][-1]
                logs.append(('batch_valid_mertric', metric))
            self.tb_logger.list_of_scalars_summary(logs, batches_done)

            self.epoch_storage['loss'].append(loss_batch)

            train_time += time.time() - curr_time
            curr_time = time.time()

        if self.debug and self.rank == 0:
            self.logger(f'loader: {loader_time:.1f} s | train: {train_time:.1f} s')

        for key, val in self.epoch_storage.items():
            if len(val) > 0:
                if isinstance(val[0], torch.Tensor):
                    self.epoch_storage[key] = torch.cat(val)
                else:
                    self.epoch_storage[key] = torch.tensor(val).to(self.device)
        
        loss_total = ((self.epoch_storage['loss'] * batch_weights).sum() / batch_weights.sum()).item()

        if self.parallel == 'ddp':
            ''' Gather tensors '''
            for key, val in self.epoch_storage.items():
                if len(val) > 0:
                    self.epoch_storage[key] = comm.gather_tensor(val)

            metric_total, monitor_metrics_total = self.evaluate_epoch(self)

        else:
            metric_total, monitor_metrics_total = self.evaluate_epoch(self)

        ''' TensorBoard logging '''
        logs = [
            ('epoch_train_loss', loss_total), 
            ('epoch_train_metric', metric_total),
        ]
        self.tb_logger.list_of_scalars_summary(logs, self.global_epoch)
        return loss_total, metric_total, monitor_metrics_total

    def _valid_one_epoch(self, loader):
        batch_weights = torch.ones(len(loader)).float().to(self.device)
        if len(loader.dataset) % loader.batch_size != 0:
            batch_weights[-1] = \
                (len(loader.dataset) % loader.batch_size)  / loader.batch_size
        # Storage
        self.epoch_storage = defaultdict(list)
        for key in ['approx', 'target', 'loss', 'batch_metric']:
            self.epoch_storage[key] = []

        self.model.eval()
        if self.progress_bar and self.rank == 0:
            iterator = enumerate(tqdm(loader, desc='valid'))
        else:
            iterator = enumerate(loader)
        
        with torch.no_grad():
            for batch_i, inputs in iterator:
                batches_done = len(loader) * (self.global_epoch-1) + batch_i
                inputs = [t.to(self.device) for t in inputs]
                loss = self.forward_train(self, inputs)

                ''' TensorBoard logging '''
                if self.parallel == 'ddp' and self.ddp_average_loss:
                    loss_batch = comm.gather_tensor(
                        loss.detach().clone().view(1)).mean().item()
                else: # Use loss on device: 0
                    loss_batch = loss.item()
                logs = [
                    ('batch_valid_loss', loss_batch),
                ]
                if len(self.epoch_storage['batch_metric']) > 0:
                    metric = self.epoch_storage['batch_metric'][-1]
                    logs.append(('batch_valid_mertric', metric))
                self.tb_logger.list_of_scalars_summary(logs, batches_done)

                self.epoch_storage['loss'].append(loss_batch)

        for key, val in self.epoch_storage.items():
            if len(val) > 0:
                if isinstance(val[0], torch.Tensor):
                    self.epoch_storage[key] = torch.cat(val)
                else:
                    self.epoch_storage[key] = torch.tensor(val).to(self.device)

        loss_total = ((self.epoch_storage['loss'] * batch_weights).sum() / batch_weights.sum()).item()
        
        if self.parallel == 'ddp':
            ''' Gather tensors '''
            for key, val in self.epoch_storage.items():
                if len(val) > 0:
                    self.epoch_storage[key] = comm.gather_tensor(val)

            metric_total, monitor_metrics_total = self.evaluate_epoch(self)

        else:
            metric_total, monitor_metrics_total = self.evaluate_epoch(self)

        ''' TensorBoard logging '''
        logs = [
            ('epoch_valid_loss', loss_total), 
            ('epoch_valid_metric', metric_total),
        ]
        self.tb_logger.list_of_scalars_summary(logs, self.global_epoch)
        return loss_total, metric_total, monitor_metrics_total

    def _train(self, loader, loader_valid, num_epochs):
        for epoch in range(num_epochs):
            if self.parallel == 'ddp':
                loader.sampler.set_epoch(epoch)
                loader_valid.sampler.set_epoch(epoch)

            ''' before train callbacks '''
            for func in self.before_train:
                func(self)

            ''' Training set '''
            loss_train, metric_train, monitor_metrics_train = \
                self._train_one_epoch(loader)

            ''' Validation set '''
            if loader_valid is None:
                loss_valid, metric_valid, monitor_metrics_valid = \
                    None, None, None
            else:
                loss_valid, metric_valid, monitor_metrics_valid = \
                    self._valid_one_epoch(loader_valid)

            self.state.update({
                'epoch': epoch, 
                'train_loss': loss_train,
                'train_metric': metric_train,
                'train_monitor': monitor_metrics_train,
                'valid_loss': loss_valid,
                'valid_metric': metric_valid,
                'valid_monitor': monitor_metrics_valid,
                'learning_rate': [group['lr'] for group in self.optimizer.param_groups][0]
            })

            if not self.batch_scheduler:  # Epoch scheduler
                if self.scheduler_target is not None:
                    self.scheduler.step(self.state[self.scheduler_target])
                else:
                    self.scheduler.step()

            ''' After train callbacks '''
            after_trains = self.after_train + [self.logger._callback]
            if self.rank != 0 and not self.debug:
                after_trains = after_trains[:-1]
            for func in after_trains:
                func(self)
            self._states.append(self.state.copy())

            if self.checkpoint and self.rank == 0:
                ''' Save model '''
                self.save_snapshot()
                self.checkpoint = False

            if self.stop_train:
                ''' Early stop '''
                if self.rank == 0:
                    self.logger('Training stopped by overfit detector.')
                break

            self.global_epoch += 1
        else:
            ''' No early stop till the end '''
            if not self.snapshot_path.exists() and self.rank == 0:
                self.save_snapshot()
        
        if self.parallel == 'ddp':
            dist.destroy_process_group()

    def _train_ddp(self, rank, dist_url, loader, loader_valid, num_epochs):
        ''' Prep for DDP '''
        set_random_seeds(0)
        self.rank = rank
        dist.init_process_group(
            backend='nccl', init_method=dist_url,
            world_size=self.world_size, rank=rank)
        comm.sync()
        torch.cuda.set_device(self.rank)
        if self.rank == 0:
            self.logger(f'All processes initialized.')

        ''' Configure model and loader '''
        self._configure_model()
        loader = self._configure_loader_ddp(loader)
        loader_valid = self._configure_loader_ddp(loader_valid, shuffle=False)

        ''' Train '''
        self._train(loader, loader_valid, num_epochs)
    
    def predict(self, loader, path=None, test_time_augmentations=1, verbose=True):
        if loader is None:
            self.logger('Skipping prediction.')
            return None

        prediction = []

        if self.progress_bar and self.rank == 0:
            iterator = tqdm(loader, desc='inference')
        else:
            iterator = loader
        self.model.eval()
        with torch.no_grad():
            for inputs in iterator:
                inputs = [t.to(self.device) for t in inputs]
                approx = self.forward_test(self, inputs)
                prediction.append(approx.detach())
        
        prediction = torch.cat(prediction).cpu().numpy()

        if path is not None:
            np.save(path, prediction)
            self.logger(f'Prediction exported to {path}')

        return prediction

    def save_snapshot(self, path=None):
        if path is None:
            path = self.snapshot_path
        if isinstance(
            self.model, 
            (torch.nn.DataParallel, DistributedDataParallel)):
            module = self.model.module
        else:
            module = self.model

        torch.save({
            'global_epoch': self.global_epoch,
            'model': module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'state': self.state, 
            'all_states': self._states
        }, path)

    def load_snapshot(self, path, device=None, 
                      load_epoch=True, load_scheduler=True):
        if device is None:
            device = self.device
        checkpoint = torch.load(path, map_location=device)
        if isinstance(
                self.model, 
                (torch.nn.DataParallel, DistributedDataParallel)):
            self.model.module.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if load_scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        if load_epoch:
            self.global_epoch = checkpoint['global_epoch']
        self.state = checkpoint['state']
        self._states = checkpoint['all_states']

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
        self.grad_accumulations = grad_accumulations
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
        self.outoffold = None
        self.prediction = None
        
        ''' Configure directory '''
        if export_dir is None:
            export_dir = Path().cwd()
        elif isinstance(export_dir, str):
            export_dir = Path(export_dir).expanduser()
        assert len(export_dir.suffix) == 0  # export_dir must be directory
        export_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_path = export_dir / f'{self.serial}.pt'

        ''' Configure loggers '''
        if self.logger is None:
            self.logger = TorchLogger(export_dir / f'{self.serial}.log')
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

        ''' Resume training '''
        if resume:
            self.load_snapshot(self.snapshot_path, 'cpu')
            self.logger(f'{self.snapshot_path} is loaded. Continuing from epoch {self.global_epoch}.')
        else:
            if self.snapshot_path.exists():
                self.snapshot_path.unlink()
        
        ''' Train '''
        self.max_epochs = self.global_epoch + num_epochs - 1
        self.dataframe = []
        self.state = {
            'train_loss': None,
            'train_metric': None,
            'train_monitor': None,
            'valid_loss': None,
            'valid_metric': None,
            'valid_monitor': None,
            'best_epoch': self.global_epoch, 
            'best_score': None, 
            'patience': 0,
            'epoch': 0, 
            'learning_rate': [group['lr'] for group in self.optimizer.param_groups][0]
        }
        self._states = []

        if self.parallel == 'ddp':
            dist_url = f'tcp://127.0.0.1:{comm.find_free_port()}'
            self.logger(f'DDP on {dist_url}')
            
            ddp_tmp = {
                'trainer': self,
                'dist_url': dist_url, 
                'loader': loader, 
                'loader_valid': loader_valid,
                'num_epochs': num_epochs
            }
            with open(DDP_TMP_PATH, 'wb') as f:
                pickle.dump(ddp_tmp, f)
            # ddp_worker_path = 'kuma_utils/torch/ddp_worker.py'
            ddp_worker_path = Path(inspect.getfile(self.__class__)).parent/'ddp_worker.py'
            ddp_procs = []
            for rank in range(self.world_size):
                command = ['python', ddp_worker_path, '--path', DDP_TMP_PATH, '--rank', str(rank)]
                proc = subprocess.Popen(command)
                ddp_procs.append(proc)
                delay = np.random.uniform(1, 5, 1)[0]
                time.sleep(delay)
            ddp_procs[0].wait()
            DDP_TMP_PATH.unlink()

            ### DDP spawn
            # mp.spawn(
            #     self._train_ddp, 
            #     nprocs=self.world_size,
            #     args=(dist_url, loader, loader_valid, num_epochs)
            # )
            ###  
            
            # !: Prediction is done by single GPU.
            # TODO: multi GPU prediction in DDP
            self.model = self.model.to(self.device)
        else:
            self._configure_model()
            self._train(loader, loader_valid, num_epochs)

        ''' Prediction '''
        self.load_snapshot(
            str(self.snapshot_path), load_epoch=False, load_scheduler=False)
        best_epoch = self.state['best_epoch']
        best_score = self.state['best_score']
        self.logger(
            f'Best epoch is [{best_epoch}], best score is [{best_score}].')

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

    def export_dataframe(self):
        return pd.DataFrame(self._states)

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
