from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import time
import pickle
import subprocess
import inspect
import os
import uuid
from pprint import pformat
import __main__
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.nn import SyncBatchNorm
from torch.utils.data.sampler import Sampler

from .utils import get_device, seed_everything, get_system_usage
from .callbacks import (
    TorchLogger, DummyLogger, SaveSnapshot
)
from .hooks import TrainHook
from . import distributed as comm
from .sampler import DistributedProxySampler
from .extras import DummyAutoCast, DummyGradScaler

try:
    from torch.cuda import amp
    AMP = True
except ModuleNotFoundError:
    AMP = False


class TorchTrainer:
    '''
    Simple Trainer for PyTorch models
    
    This is something similar to PyTorch Lightning, but this works with vanilla PyTorch modules.
    '''

    def __init__(self,
                 model, device=None, serial='exp00'):

        self.serial = serial
        self.device, self.device_ids = get_device(device)
        self.world_size = len(self.device_ids)
        self.model = model
        self.rank = 0
        self._register_ready = False
        self._model_ready = False
        self.logger = None

        # Implicit attributes
        # DDP
        self.ddp_sync_batch_norm = SyncBatchNorm.convert_sync_batchnorm
        self.ddp_average_loss = True
        self.ddp_params = dict(
            broadcast_buffers=True,
            static_graph=True,
            # find_unused_parameters=True
        )
        self.ddp_workers = -1
        # MISC
        self.loader_to_callback = False
        self.debug = False
        self.display_ett_time = 30
        self.fix_nan = False

    def _register_callbacks(self, callbacks):
        self.before_epoch = [func.before_epoch for func in callbacks]
        self.after_epoch = [func.after_epoch for func in callbacks]
        self._save_snapshot = [func.save_snapshot for func in callbacks]
        self._load_snapshot = [func.load_snapshot for func in callbacks]

    def _register_hook(self, hook):
        self.forward_train = hook.forward_train
        self.forward_valid = hook.forward_valid
        self.forward_test = hook.forward_test
        self.backprop = hook.backprop
        self.evaluate_batch = hook.evaluate_batch
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
        if self.parallel == 'dp':  # DP on cuda
            self.model = DataParallel(
                self.model, device_ids=self.device_ids).to(self.device)
            if hasattr(self, 'criterion') and self.criterion is not None:
                self.criterion = self.criterion.to(self.device)
            self.logger(f'DataParallel on device: {self.device_ids}')

        elif self.parallel == 'ddp':  # DDP on cuda
            self.model = self.ddp_sync_batch_norm(self.model)
            self.model = DistributedDataParallel(
                self.model.to(self.rank), device_ids=[self.rank],
                **self.ddp_params
            )
            if hasattr(self, 'criterion') and self.criterion is not None:
                self.criterion = self.criterion.to(self.rank)
            if self.rank == 0:
                self.logger(
                    f'DistributedDataParallel on device: {self.device_ids}')

        elif self.parallel is not None:
            raise ValueError(f'Unknown type of parallel {self.parallel}')

        else:  # Single device
            self.model.to(self.device)
            if hasattr(self, 'criterion') and self.criterion is not None:
                self.criterion = self.criterion.to(self.device)
            self.logger(f'Model on device: {self.device}')
        
        self._model_ready = True

    def _configure_loader_ddp(self, loader, shuffle=True):
        if loader is None:
            return None
        skip_keys = ['sampler', 'batch_sampler', 'dataset_kind']
        dl_args = {
            k: v for k, v in loader.__dict__.items()
            if not k.startswith('_') and k not in skip_keys
        }
        if isinstance(loader.sampler, Sampler):
            sampler = DistributedProxySampler(
                loader.sampler, num_replicas=self.world_size, rank=self.rank)
        else:
            sampler = DistributedSampler(
                loader.dataset, num_replicas=self.world_size, rank=self.rank, shuffle=shuffle)
        dl_args['sampler'] = sampler
        if self.ddp_workers == -1:
            dl_args['num_workers'] = int(
                dl_args['num_workers'] / self.world_size)
        else:
            dl_args['num_workers'] = self.ddp_workers
        if dl_args['batch_size'] % self.world_size != 0:
            raise ValueError(
                f'batch size must be a multiple of world size({self.world_size}).')
        dl_args['batch_size'] = int(dl_args['batch_size'] / self.world_size)
        return type(loader)(**dl_args)

    def _find_and_fix_nan(self, inputs, loss, approx, prefix=''):
        if torch.isnan(loss).any():
            if self.rank == 0:
                self.logger(f'{prefix} {torch.isnan(loss).sum()} NaN detected in loss.')
            loss = torch.nan_to_num(loss)
            for input_i, input_t in enumerate(inputs):
                if torch.isnan(input_t).any():
                    if self.rank == 0:
                        self.logger(f'{prefix} NaN detected in {input_i}-th input.')
        if torch.isnan(approx).any():
            if self.rank == 0:
                self.logger(f'{prefix} {torch.isnan(approx).sum()} NaN detected in output.')
            approx = torch.nan_to_num(approx)
        return loss, approx

    def _gather_storage(self):
        for key, val in self.epoch_storage.items():
            if len(val) > 0:
                self.epoch_storage[key] = torch.nan_to_num(comm.gather_tensor(val))
                if self.debug:
                    self.logger(f'[rank {self.rank}] gather storage {key}: {self.epoch_storage[key].shape}')

    def _concat_storage(self):
        for key, val in self.epoch_storage.items():
            if len(val) > 0:
                if isinstance(val[0], torch.Tensor):  # val: [(batch, ...), (batch, ...), ...]
                    self.epoch_storage[key] = torch.nan_to_num(torch.cat(val))
                else:  # val: [value, value, ...]
                    self.epoch_storage[key] = torch.nan_to_num(torch.tensor(val)).to(self.device)
                if self.debug:
                    self.logger(f'[rank {self.rank}] concat storage {key}: {self.epoch_storage[key].shape}')

    def _train_one_epoch(self, loader):
        loader_time = .0
        train_time = .0
        start_time = time.time()
        curr_time = time.time()

        self.epoch_storage = defaultdict(list)
        for key in ['approx', 'target', 'loss', 'batch_metric']:
            self.epoch_storage[key] = []

        self.model.train()
        if self.progress_bar and self.rank == 0:
            iterator = enumerate(tqdm(loader, desc='train'))
        else:
            iterator = enumerate(loader)
        batch_total = len(loader)
        ett_disp = False

        for batch_i, inputs in iterator:
            loader_time += time.time() - curr_time
            curr_time = time.time()
            elapsed_time = curr_time - start_time
            if self.rank == 0 and self.state['epoch'] == 0 and elapsed_time > self.display_ett_time and not ett_disp:  # show ETA
                ett = elapsed_time * batch_total // (batch_i + 1)
                system_usage = get_system_usage()
                self.logger(f'Estimated epoch training time: {int(ett)} s')
                self.logger(f'Maximum RAM usage: {system_usage["ram_usage"]} MB')
                self.logger(f'Maximum GRAM usage: {system_usage["gram_usage"]}')
                ett_disp = True

            batches_done = batch_total * (self.global_epoch-1) + batch_i
            inputs = [t.to(self.device) for t in inputs]

            # forward and backward
            with self.autocast:
                loss, approx = self.forward_train(self, inputs)
                self.evaluate_batch(self, inputs, approx)
            loss = loss / self.grad_accumulations
            if ((batch_i + 1) % self.grad_accumulations == 0) or ((batch_i + 1) == len(iterator)):
                self.backprop(self, loss, inputs)
            if self.batch_scheduler:
                self.scheduler.step()

            # detect nan value
            if self.fix_nan:
                loss, approx = self._find_and_fix_nan(
                    inputs, loss, approx,
                    prefix=f'[{self.rank}] ({batch_i}/{len(loader)})')

            # logging
            if self.parallel == 'ddp' and self.ddp_average_loss:
                loss_batch = comm.gather_tensor(loss.detach().clone().view(1)).mean().item()
            else:  # Use loss on device: 0
                loss_batch = loss.item()

            learning_rate = [param_group['lr']
                             for param_group in self.optimizer.param_groups]
            logs = {
                'batch_train_loss': loss_batch,
                'batch_train_lr': learning_rate[0]
            }
            if len(self.epoch_storage['batch_metric']) > 0:
                metric = self.epoch_storage['batch_metric'][-1]
                logs['batch_valid_metric'] = metric
            if self.rank == 0:
                self.logger.write_log(logs, batches_done, log_wandb=False)
            self.epoch_storage['loss'].append(loss_batch)

            train_time += time.time() - curr_time
            curr_time = time.time()

        self._concat_storage()
        loss_total = self.epoch_storage['loss'].mean().item()

        if self.parallel == 'ddp':
            self._gather_storage()
            metric_total, monitor_metrics_total = self.evaluate_epoch(self)
        else:
            metric_total, monitor_metrics_total = self.evaluate_epoch(self)

        if self.eval_metric is None:
            metric_total = loss_total

        if self.debug:
            self.logger(f'[rank {self.rank}] loader: {loader_time:.1f} s | train: {train_time:.1f} s')

        return loss_total, metric_total, monitor_metrics_total

    def _valid_one_epoch(self, loader):
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
                loss, approx = self.forward_valid(self, inputs)
                if self.fix_nan:
                    loss, approx = self._find_and_fix_nan(
                        inputs, loss, approx,
                        prefix=f'[{self.rank}] ({batch_i}/{len(loader)})')
                self.evaluate_batch(self, inputs, approx)
                if self.parallel == 'ddp' and self.ddp_average_loss:
                    loss_batch = comm.gather_tensor(
                        loss.detach().clone().view(1)).mean().item()
                else:  # Use loss on device: 0
                    loss_batch = loss.item()

                logs = {
                    'batch_valid_loss': loss_batch
                }
                if len(self.epoch_storage['batch_metric']) > 0:
                    metric = self.epoch_storage['batch_metric'][-1]
                    logs['batch_valid_metric'] = metric
                if self.rank == 0:
                    self.logger.write_log(logs, batches_done, log_wandb=False)
                self.epoch_storage['loss'].append(loss_batch)

        self._concat_storage()
        loss_total = self.epoch_storage['loss'].mean().item()

        if self.parallel == 'ddp':
            self._gather_storage()
            metric_total, monitor_metrics_total = self.evaluate_epoch(self)
        else:
            metric_total, monitor_metrics_total = self.evaluate_epoch(self)

        if self.eval_metric is None:
            metric_total = loss_total

        return loss_total, metric_total, monitor_metrics_total

    def _train(self, loader, loader_valid, num_epochs):
        assert self._register_ready
        assert self._model_ready
        if self.rank == 0 and hasattr(self.logger, 'use_wandb') and self.logger.use_wandb:
            self.logger.init_wandb(serial=self.serial)
        if self.rank == 0 and hasattr(self.logger, 'use_tensorboard') and self.logger.use_tensorboard:
            self.logger.init_tensorboard(serial=self.serial)
        if self.fp16:
            self.scaler = amp.GradScaler()
            self.autocast = amp.autocast()
        else:
            self.scaler = DummyGradScaler()
            self.autocast = DummyAutoCast()
        
        for epoch in range(num_epochs):
            if self.parallel == 'ddp':
                loader.sampler.set_epoch(epoch)
            
            self.state.update({'epoch': epoch})

            ''' before epoch callbacks '''
            for func in self.before_epoch:
                if self.loader_to_callback:
                    func(self, loader, loader_valid)
                else:
                    func(self)

            ''' Training loop '''
            loss_train, metric_train, monitor_metrics_train = \
                self._train_one_epoch(loader)

            ''' Validation loop '''
            if loader_valid is None:
                loss_valid, metric_valid, monitor_metrics_valid = \
                    None, None, None
            else:
                if self.parallel == 'ddp':
                    loader_valid.sampler.set_epoch(epoch)
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

            ''' After epoch callbacks '''
            after_trains = self.after_epoch + [self.logger.after_epoch]
            if self.rank != 0:  # export logs on rank 0 device only
                after_trains = after_trains[:-1]
            for func in after_trains:
                if self.loader_to_callback:
                    func(self, loader, loader_valid)
                else:
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

        if self.parallel == 'ddp':
            dist.destroy_process_group()

    def _train_ddp(self, rank, dist_url, loader, loader_valid, num_epochs):
        seed_everything(self.random_state, self.deterministic)
        self.rank = rank
        dist.init_process_group(
            backend='nccl', init_method=dist_url,
            world_size=self.world_size, rank=rank)
        comm.sync()
        torch.cuda.set_device(self.rank)
        if self.rank == 0:
            self.ddp_tmp_path.unlink()
            self.logger('All processes initialized.')
        
        ''' Configure model and loader '''
        self._configure_model()
        loader = self._configure_loader_ddp(loader)
        loader_valid = self._configure_loader_ddp(loader_valid, shuffle=False)

        ''' Train '''
        self._train(loader, loader_valid, num_epochs)

    def predict(self, loader, parallel=None, fp16=False, progress_bar=False):
        self.parallel = parallel
        if self.logger is None:
            self.logger = DummyLogger('')
        if not self._register_ready:  # is hook and callbacks registered?
            raise AttributeError('Register hook and callbacks by .register() method.')
        if not self._model_ready:  # is model configured?
            self.fp16 = fp16
            if parallel == 'ddp':
                raise NotImplementedError('DDP prediction is not implemented.')
            else:
                self._configure_model()
                
        if progress_bar:
            iterator = tqdm(loader, desc='inference')
        else:
            iterator = loader
        prediction = []
        self.model.eval()
        with torch.no_grad():
            for inputs in iterator:
                inputs = [t.to(self.device) for t in inputs]
                if self.fp16:
                    with amp.autocast():
                        approx = self.forward_test(self, inputs)
                else:
                    approx = self.forward_test(self, inputs)
                prediction.append(approx.detach())
        prediction = torch.cat(prediction).float().cpu().numpy()

        return prediction

    def save_snapshot(self, path=None):
        for func in self._save_snapshot:
            func(self, path)

    def load_snapshot(self, path=None, device=None):
        for func in self._load_snapshot:
            func(self, path, device)

    def register(self, hook=TrainHook(), callbacks=[SaveSnapshot()]):
        # This function must be called
        self._register_hook(hook)
        self._register_callbacks(callbacks)
        self._register_ready = True

    def train(self,
              # Essential
              criterion, optimizer, scheduler, loader, num_epochs,
              batch_scheduler=False, scheduler_target=None,
              hook=TrainHook(), callbacks=[SaveSnapshot()],
              # Evaluation
              loader_valid=None, eval_metric=None, monitor_metrics=[],
              # Snapshot
              export_dir=None, resume=False,
              # Training option
              fp16=False, parallel=None, grad_accumulations=1,
              deterministic=None, random_state=0,
              # Logging
              logger=None, progress_bar=False,
              **kw_args
              ):
        # Register params
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_scheduler = batch_scheduler
        self.scheduler_target = scheduler_target
        self.grad_accumulations = grad_accumulations
        self.deterministic = deterministic
        self.random_state = random_state
        self.eval_metric = eval_metric
        self.monitor_metrics = monitor_metrics
        self.logger = logger
        self.fp16 = fp16
        self.parallel = parallel
        self.progress_bar = progress_bar
        self.register(hook=hook, callbacks=callbacks)

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
        self.base_dir = export_dir
        self.snapshot_path = self.base_dir / f'{self.serial}.pt' 

        ''' Configure loggers '''
        if self.logger is None:
            self.logger = TorchLogger(self.base_dir / f'{self.serial}.log')
        elif isinstance(self.logger, (str, Path)):
            self.logger = TorchLogger(self.logger, file=True)
        elif isinstance(self.logger, TorchLogger):
            pass
        else:
            raise ValueError('Invalid type of logger.')
        if len(kw_args) > 0:
            self.logger(f'{kw_args} will be ignored.')

        ''' Configure loss function and metrics '''
        if criterion is None:
            self.logger('criterion is not set. Make sure loss is calculated in the training hook.')
        if eval_metric is None:
            self.logger('eval_metric is not set. criterion will be used.')
        if not isinstance(self.monitor_metrics, (list, tuple)):
            self.monitor_metrics = [self.monitor_metrics]

        ''' Resume training '''
        if resume:
            self.load_snapshot(self.snapshot_path, device='cpu')
            self.global_epoch += 1
            self.logger(f'Continuing from epoch {self.global_epoch}.')

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
            session_id = str(uuid.uuid4())
            origin = Path.cwd() / __main__.__file__
            self.logger(f'DDP URL :\t{dist_url}')
            self.logger(f'session id :\t{session_id}')
            self.logger(f'__main__ :\t{origin}')
            
            ddp_tmp = {
                'trainer': self,
                'dist_url': dist_url,
                'loader': loader,
                'loader_valid': loader_valid,
                'num_epochs': num_epochs
            }
            ddp_tmp_path = Path(f'.ku_ddp_tmp_{session_id}')
            self.ddp_tmp_path = ddp_tmp_path
            with open(ddp_tmp_path, 'wb') as f:
                pickle.dump(ddp_tmp, f)
            ddp_worker_path = Path(inspect.getfile(
                self.__class__)).parent/'ddp_worker.py'
            env_copy = os.environ.copy()
            env_copy['OMP_NUM_THREADS'] = '1'
            
            command = [
                'torchrun',
                '--standalone',
                '--nnodes', '1',
                '--nproc_per_node', str(self.world_size), 
                '--rdzv_endpoint', dist_url,
                ddp_worker_path,
                '--path', ddp_tmp_path,
                '--origin', str(origin)
            ]
            proc = subprocess.Popen(
                command, env=env_copy, cwd=origin.parent)
            proc.wait()
            if ddp_tmp_path.exists():
                ddp_tmp_path.unlink()
        else:
            self._configure_model()
            self._train(loader, loader_valid, num_epochs)

    fit = train  # for compatibility
    load_checkpoint = load_snapshot
    save_checkpoint = save_snapshot

    def export_dataframe(self):
        return pd.DataFrame(self._states)

    def __repr__(self):
        print_dict = {
            'model': self.model.__class__.__name__,
            'device': self.device,
            'serial': self.serial
        }
        return f'TorchTrainer(\n{pformat(print_dict, compact=True, indent=2)})'
