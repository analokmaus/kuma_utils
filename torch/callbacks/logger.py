import time
from pathlib import Path


def get_time(time_format='%H:%M:%S'):
    return time.strftime(time_format, time.gmtime())


class TorchLogger:

    def __init__(self, path,
                 log_items=[
                     'epoch', 'train_loss', 'valid_loss', 'train_metric', 'valid_metric',
                     'train_monitor', 'valid_monitor', 'learning_rate', 'patience'],
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
                log_str += f'[Epoch {env.trainer.global_epoch:-3}/{env.trainer.max_epochs:-3}] '
            elif item == 'patience':
                if env.trainer.patience > 0:
                    log_str += f'(*{env.trainer.patience})'
            else:
                val = env.results[item]
                if val is None:
                    continue
                elif isinstance(val, list):
                    metrics_str = '[' + \
                        ', '.join([f'{v:.6f}' for v in val]) + ']'
                    if len(val) > 0:
                        log_str += f"{item}={metrics_str} | "
                else:
                    log_str += f"{item}={val:.6f} | "
        if len(log_str) > 0:
            log_str = f'{get_time()} ' + log_str
        if self.stdout:
            print(log_str)
        if self.file:
            with open(self.path, 'a') as f:
                f.write(log_str + '\n')
