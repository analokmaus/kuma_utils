'''
Stopper

# Methods
__call__(score) : bool  = return whether score is improved
stop() : bool           = return whether to stop training or not
state() : int, int      = return current / total
score() : float         = return best score
freeze()                = update score but never stop
unfreeze()              = unset freeze()
'''
import numpy as np


class DummyStopper:
    ''' No stopper '''

    def __init__(self):
        pass

    def __call__(self, val_loss):
        return True

    def stop(self):
        return False

    def state(self):
        return 0, 0

    def score(self):
        return 0.0

    def dump_state_dict(self):
        return {}

    def load_state_dict(self, checkpoint):
        pass

    def __repr__(self):
        return 'No Stopper'


class EarlyStopping(DummyStopper):
    '''
    Early stops the training if validation loss doesn't improve after a given patience.
    patience: int   = early stopping rounds
    maximize: bool  = whether maximize or minimize metric
    '''

    def __init__(self, patience=5, maximize=False):
        super().__init__()
        self.patience = patience
        self.counter = 0
        self.log = []
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        if maximize:
            self.coef = 1
        else:
            self.coef = -1
        self.frozen = False

    def __call__(self, val_loss):
        score = self.coef * val_loss
        self.log.append(score)
        if self.best_score is None:
            self.best_score = score
            return True
        elif score < self.best_score:
            if not self.frozen:
                self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:  # score improved
            self.best_score = score
            self.counter = 0
            return True

    def stop(self):
        return self.early_stop

    def state(self):
        return self.counter, self.patience

    def score(self):
        return self.best_score

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    def reset(self):
        self.best_score = None

    def dump_state_dict(self):
        return {
            'best_score': self.best_score,
            'counter': self.counter,
        }

    def load_state_dict(self, checkpoint):
        self.best_score = checkpoint['best_score']
        self.counter = checkpoint['counter']

    def __repr__(self):
        return f'EarlyStopping({self.patience})'
