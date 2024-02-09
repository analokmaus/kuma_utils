try:
    import wandb
    WANDB = True
except:
    WANDB = False
    pass

from kuma_utils.torch.utils import get_gpu_memory, get_time


class TorchLogger:

    def __init__(self, path,
                 log_items=[
                     'epoch', 'train_loss', 'valid_loss', 'train_metric', 'valid_metric',
                     'train_monitor', 'valid_monitor', 'learning_rate', 'early_stop'],
                 verbose_eval=1,
                 stdout=True, file=False, 
                 use_wandb=False, wandb_params={'project_name': 'test', 'config': {}}):
        if isinstance(log_items, str):
            log_items = log_items.split(' ')
        self.path = path
        self.log_items = log_items
        self.verbose_eval = verbose_eval
        self.stdout = stdout
        self.file = file
        self.use_wandb = use_wandb
        self.wandb_params = wandb_params
        self.sep = ' | '
        log_str = f'TorchLogger created at {get_time("%y/%m/%d %H:%M:%S")}'
        if self.stdout:
            print(log_str)
        if self.file:
            with open(self.path, 'w') as f:
                f.write(log_str + '\n')
        self.dataframe = []

    def init_wandb(self): # This is called in Trainer._train()
        if not WANDB:
            raise ValueError('wandb is not installed.')
        wandb.init(**self.wandb_params)

    def __call__(self, log_str):
        log_str = get_time() + ' ' + log_str
        if self.stdout:
            print(log_str)
        if self.file:
            with open(self.path, 'a') as f:
                f.write(log_str + '\n')

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
                log_dict.update({'early_stopping_counter': counter, 'best_score': best_score})
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
        if len(log_str) > 0:
            log_str = f'{get_time()} ' + log_str
        if self.stdout:
            print(log_str)
        if self.file:
            with open(self.path, 'a') as f:
                f.write(log_str + '\n')
        if self.use_wandb:
            wandb.log(log_dict, step=epoch)


class DummyLogger:

    def __init__(self, path):
        pass

    def __call__(self, log_str):
        pass

    def after_epoch(self, env):
        pass
