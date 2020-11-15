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
        for dataset, metric_name, metric, _ in env.evaluation_result_list:
            log_str += f'{dataset} {metric_name} {metric:.6f} / '
        log_str += '\n'
        with open(self.path, 'a') as f:
            f.write(log_str)


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
