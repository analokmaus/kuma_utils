import time
import logging
from pathlib import Path
from pprint import pprint, pformat


def get_time(time_format='%H:%M:%S'):
    return time.strftime(time_format, time.gmtime())


class LGBMLogger:

    def __init__(
            self,
            path: str | Path,
            stdout: bool = True,
            file: bool = False,
            logger_name: str = 'LGBMLogger',
            default_level: str = 'INFO'):
        self.path = path
        self.stdout = stdout
        self.file = file
        self.logger_name = logger_name
        self.level = default_level
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

    def lgbm(self, env):
        log_str = ''
        log_str += f'[iter {env.iteration:-5}] '
        for inputs in env.evaluation_result_list:
            for i in inputs:
                if isinstance(i, str):
                    log_str += f'{i} '
                elif isinstance(i, bool):
                    pass
                else:
                    log_str += f'{i:.6f} '
        else:
            log_str += '/ '
        log_str += '\n'
        self.debug(log_str)

    def optuna(self, study, trial):
        best_score = study.best_value
        curr_score = trial.value
        if curr_score == best_score:
            log_str = ''
            log_str += f'[trial {trial.number:-4}] New best: {best_score:.6f} \n'
            log_str += f'{pformat(study.best_params, compact=True, indent=2)}'
            self.info(log_str)

    def __call__(self, log_str):
        self.info(log_str)
