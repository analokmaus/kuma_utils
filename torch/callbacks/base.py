from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CallbackEnv:
    trainer: Any = None
    epoch: int = None
    results: dict = field(default_factory=dict)


class CallbackTemplate:
    '''
    Callback Template for TorchTrainer
    '''

    def __init__(self):
        pass

    def before_train(self, env):
        pass

    def after_train(self, env):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, checkpoint):
        pass

    def __repr__(self):
        return self.__class__.__name__
