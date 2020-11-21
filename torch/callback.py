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
    best_epoch: Any = None
    score: Any = None
    best_score: Any = None
    loss_train: Any = None
    loss_valid: Any = None
    metric_train: Any = None
    metric_valid: Any = None
    monitor_metrics_train: Any = None
    monitor_metrics_valid: Any = None
    logger: Any = None


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
        env.logger.early_stop_counter = self.state['counter']
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
                env.logger._callback(env)
                raise EarlyStoppingTrigger()
        
    def state_dict(self):
        return self.state

    def load_state_dict(self, checkpoint):
        self.state = checkpoint

    def __repr__(self):
        return f'EarlyStopping({self.state})'


class TorchLogger:

    def __init__(self, path, 
                 log_items=[
                     'epoch', 'loss_train', 'loss_valid', 'metric_train', 'metric_valid', 
                     'monitor_metrics_train', 'monitor_metrics_valid', 'earlystop'], 
                 verbose_eval=1,
                 stdout=True, file=False):
        self.path = path
        self.log_items = log_items
        self.verbose_eval = verbose_eval
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
        if env.epoch % self.verbose_eval != 0:
            return
        log_str = ''
        for item in self.log_items:
            if item == 'epoch':
                log_str += f'[Epoch {env.global_epoch:-3}/{env.max_epoch:-3}] '
            elif item == 'earlystop':
                counter = env.global_epoch-env.best_epoch
                if counter > 0:
                    log_str += f'(*{counter})'
            else:
                val = getattr(env, item)
                if val is None:
                    continue
                elif isinstance(val, list):
                    metrics_str = '[' + ', '.join([f'{v:.6f}' for v in val]) + ']'
                    if len(val) > 0:
                        log_str += f"{item}={metrics_str} "
                else:
                    log_str += f"{item}={val:.6f} | "
        if len(log_str) > 0:
            log_str = f'{get_time()} ' + log_str
        if self.stdout:
            print(log_str)
