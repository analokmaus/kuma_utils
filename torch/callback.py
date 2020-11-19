from dataclasses import dataclass
import time
from typing import Any


@dataclass(frozen=True)
class CallbackEnv:
    serial: Any = None
    model: Any = None
    optimizer: Any = None
    scheduler: Any = None
    criterion: Any = None
    eval_metric: Any = None
    epoch: Any = None
    global_epoch: Any = None
    max_epoch: Any = None
    score: Any = None
    loss_train: Any = None
    loss_valid: Any = None
    metric_train: Any = None
    metric_valid: Any = None
    monitor_metrics_train: Any = None
    monitor_metrics_valid: Any = None


class SaveModelTrigger(Exception): pass
class EarlyStoppingTrigger(Exception): pass


def get_time(time_format='%H:%M:%S'):
    return time.strftime(time_format, time.gmtime())


class CallbackTemplate:
    '''

    '''

    def __init__(self):
        pass

    def __call__(self, env):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, checkpoint):
        pass

    def __repr__(self):
        return self.__class__.__name__


class EarlyStopping(CallbackTemplate):
    '''
    Early stops the training if validation loss doesn't improve after a given patience.
    patience: int       = early stopping rounds
    maximize: bool      = whether maximize or minimize metric
    start_epoch: int    = 
    '''

    def __init__(self, patience=5, maximize=False, skip_epoch=0):
        super().__init__()
        self.state = {
            'patience': patience,
            'maximize': maximize,
            'skip_epoch': skip_epoch, 
            'counter': 0, 
            'best_score': None
        }

    def __call__(self, env):
        score = env.score
        if env.epoch <= self.state['skip_epoch']:
            self.state['best_score'] = score
            raise SaveModelTrigger()
        else:
            if (self.state['maximize'] and score > self.state['best_score']) or \
                    (not self.state['maximize'] and score < self.state['best_score']):
                self.state['best_score'] = score
                self.state['counter'] = 0
                raise SaveModelTrigger()
            else:
                self.state['counter'] += 1
            
            if self.state['counter'] >= self.state['patience']:
                raise EarlyStoppingTrigger()
        
    def state_dict(self):
        return self.state

    def load_state_dict(self, checkpoint):
        self.state = checkpoint

    def __repr__(self):
        return f'EarlyStopping({self.state})'


class TorchLogger:

    def __init__(self, path, stdout=True, file=False):
        self.path = path
        self.stdout = stdout
        self.file = file
        log_str = f'TorchLogger created at {get_time("%y/%m/%d:%H:%M:%S")}'
        if self.stdout:
            print(log_str)
        if self.file:
            with open(self.path, 'w') as f:
                f.write(log_str + '\n')

    def __call__(self, log_str):
        log_str = get_time() + ' ' + log_str
        if self.stdout:
            print(log_str)
        if self.file:
            with open(self.path, 'a') as f:
                f.write(log_str + '\n')

    def _callback(self, env):
        log_str = f'[{env.serial}] {get_time()} '
        log_str += f'[{env.global_epoch:-3}/{env.max_epoch:-3}] '
        for item in [
            'loss_train', 'loss_valid', 'metric_train', 
            'metric_valid', 'monitor_metrics_valid']:
            val = getattr(env, item)
            if val is None:
                continue
            elif isinstance(val, list):
                log_str += f"{item} = {val} "
            else:
                log_str += f"{item} = {val:.6f} "
        if self.stdout:
            print(log_str)
