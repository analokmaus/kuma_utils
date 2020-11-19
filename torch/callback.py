from collections import namedtuple


CallbackEnv = namedtuple(
    'TorchTrainerEnv', [
        'model', 'optimizer', 'scheduler', 'stopper', 'criterion', 
        'eval_metric', 'epoch', 'global_epoch', 'log'
    ]
)


class DummyEvent:
    ''' Dummy event does nothing '''

    def __init__(self):
        pass

    def __call__(self, env):
        pass

    def dump_state_dict(self):
        return {}

    def load_state_dict(self, checkpoint):
        pass

    def __repr__(self):
        return 'No Event'


class NoEarlyStoppingNEpochs(DummyEvent):

    def __init__(self, n):
        super().__init__()
        self.n = n

    def __call__(self, env):
        if env.global_epoch == 0:
            env.stopper.freeze()
            env.stopper.reset()
            print(f"Epoch\t{env.epoch}: Earlystopping is frozen.")
        elif env.global_epoch < self.n:
            env.stopper.reset()
        elif env.global_epoch == self.n:
            env.stopper.unfreeze()
            print(f"Epoch\t{env.epoch}: Earlystopping is unfrozen.")

    def __repr__(self):
        return f'NoEarlyStoppingNEpochs({self.n})'
