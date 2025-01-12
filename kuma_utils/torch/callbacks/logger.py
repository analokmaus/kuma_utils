import logging
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
try:
    import wandb
    WANDB = True
except Exception as e:
    WANDB = False
    pass

from kuma_utils.torch.utils import get_gpu_memory, get_time


class TorchLogger:

    def __init__(
            self,
            path: str | Path,
            log_items: list[str] | str = [
                'epoch',
                'train_loss', 'valid_loss',
                'train_metric', 'valid_metric',
                'train_monitor', 'valid_monitor',
                'learning_rate', 'early_stop'],
            verbose_eval: int = 1,
            stdout: bool = True,
            file: bool = False,
            logger_name: str = 'TorchLogger',
            default_level: str = 'INFO',
            use_wandb: bool = False,
            wandb_params: dict = {'project': 'test', 'config': {}},
            use_tensorboard: bool = False,
            tensorboard_dir: str | Path = None):
        if isinstance(log_items, str):
            log_items = log_items.split(' ')
        self.path = path
        if isinstance(self.path, str):
            self.path = Path(self.path)
        self.log_items = log_items
        self.verbose_eval = verbose_eval
        self.stdout = stdout
        self.file = file
        self.logger_name = logger_name
        self.level = default_level
        self.use_wandb = use_wandb
        self.wandb_params = wandb_params
        self.use_tensorboard = use_tensorboard
        self.tensorboard_dir = tensorboard_dir
        self.sep = ' | '
        self.dataframe = []

        self.system_logger = logging.getLogger(self.logger_name)
        for handler in self.system_logger.handlers[:]:
            self.system_logger.removeHandler(handler)
            handler.close()
        self.system_logger.setLevel(self.level)
        formatter = logging.Formatter("%(asctime)s - %(levelname)-8s - %(message)s")
        if self.file:
            fh = logging.FileHandler(self.path)
            fh.setFormatter(formatter)
            self.system_logger.addHandler(fh)
        if self.stdout:
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            self.system_logger.addHandler(sh)
        for level in ['debug', 'info', 'warning', 'error', 'critical']:
            setattr(self, level, getattr(self.system_logger, level))

    def init_wandb(self, serial: str = None):  # This is called in Trainer._train()
        if not WANDB:
            raise ValueError('wandb is not installed.')
        wandb_params = self.wandb_params.copy()
        if serial is not None and 'name' not in wandb_params.keys():  # No override
            wandb_params.update({'name': serial})
        wandb.init(**wandb_params)

    def init_tensorboard(self, serial):  # This is called in Trainer._train()
        if self.tensorboard_dir is None and self.path is not None:
            self.tensorboard_dir = self.path.parent/'tensorboard'
        self.tensorboard_dir = self.tensorboard_dir/serial
        if not self.tensorboard_dir.exists():
            self.tensorboard_dir.mkdir(exist_ok=True, parents=True)
        self.tb_writer = SummaryWriter(log_dir=self.tensorboard_dir)

    def __call__(self, log_str):
        self.info(log_str)

    def after_epoch(self, env, loader=None, loader_valid=None):
        ''' callback '''
        epoch = env.state['epoch']
        if epoch % self.verbose_eval != 0:
            return
        log_str = ''
        log_dict = {}
        for item in self.log_items:
            if item == 'epoch':
                num_len = len(str(env.max_epochs))
                log_str += f'Epoch {env.global_epoch:-{num_len}}/'
                log_str += f'{env.max_epochs:-{num_len}}'
                log_dict['global_epoch'] = env.global_epoch
            elif item == 'early_stop':
                best_score = env.state['best_score']
                counter = env.state['patience']
                if counter > 0:
                    log_str += f'best={best_score:.6f}(*{counter})'
                log_dict.update({
                    'early_stopping_counter': counter,
                    'best_score': best_score})
            elif item == 'gpu_memory':
                log_str += 'gpu_mem='
                for gpu_i, gpu_mem in get_gpu_memory().items():
                    log_str += f'({gpu_i}:{int(gpu_mem)}MB)'
            else:
                val = env.state[item]
                if val is None:
                    continue
                elif isinstance(val, list):
                    metrics_str = '[' + \
                        ', '.join([f'{v:.6f}' for v in val]) + ']'
                    if len(val) > 0:
                        log_str += f"{item}={metrics_str}"
                    for iv, v in enumerate(val):
                        log_dict[f'{item}{iv}'] = v
                else:
                    log_str += f"{item}={val:.6f}"
                    log_dict[item] = val
            log_str += self.sep
        self.__call__(log_str)
        self.write_log(log_dict, epoch)

    def write_log(self,
                  logs: dict,
                  step: int,
                  log_wandb: bool = True,
                  log_tensorboard: bool = True):
        if self.use_wandb and log_wandb:
            wandb.log(logs, step=step)
        if self.use_tensorboard and log_tensorboard:
            for k, v in logs.items():
                self.tb_writer.add_scalar(k, v, step)


class DummyLogger:

    def __init__(self, path):
        for level in ['debug', 'info', 'warning', 'error', 'critical']:
            setattr(self, level, self.__call__)

    def __call__(self, log_str):
        pass

    def after_epoch(self, env):
        pass

    def write_log(self, logs, step):
        pass
