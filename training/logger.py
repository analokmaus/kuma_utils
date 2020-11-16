import time


def get_time(time_format='%H:%M:%S'):
    return time.strftime(time_format, time.gmtime())


class LGBMLogger:

    def __init__(self, path, params={}, fit_params={}):
        self.path = path
        log_str = f'Logger created at {get_time("%y/%m/%d:%H:%M:%S")}\n'
        log_str += f'params: {params}\n'
        log_str += f'fit_params: {fit_params}\n'
        with open(self.path, 'w') as f:
            f.write(log_str)
        
    def __call__(self, env):
        log_str = f'{get_time()} '
        log_str += f'[iter {env.iteration:05}] '
        log_type = 'lgb' if len(env.evaluation_result_list[0]) == 4 else 'xgb'
        if log_type == 'lgb':
            for dataset, metric_name, metric, _ in env.evaluation_result_list:
                log_str += f'{dataset} {metric_name} {metric:.6f} / '
        elif log_type == 'xgb':
            for key, metric in env.evaluation_result_list:
                log_str += f'{key} {metric:.6f} / '
        log_str += '\n'
        with open(self.path, 'a') as f:
            f.write(log_str)

    def write(self, text):
        text = get_time() + ' ' + text
        with open(self.path, 'a') as f:
            f.write(text)


class XGBLogger:

    def __init__(self, path, params={}, fit_params={}):
        self.path = path
        log_str = f'Logger created at {get_time("%y/%m/%d:%H:%M:%S")}\n'
        log_str += f'params: {params}\n'
        log_str += f'fit_params: {fit_params}\n'
        with open(self.path, 'w') as f:
            f.write(log_str)

    def __call__(self, env):
        log_str = f'{get_time()} '
        log_str += f'[iter {env.iteration:05}] '
        for key, metric in env.evaluation_result_list:
            log_str += f'{key} {metric:.6f} / '
        log_str += '\n'
        with open(self.path, 'a') as f:
            f.write(log_str)
