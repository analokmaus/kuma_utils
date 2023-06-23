from .base import CallbackTemplate
from pprint import pformat
import numpy as np


class SaveEveryEpoch(CallbackTemplate):
    '''
    Save snapshot every epoch
    '''

    def __init__(self, ):
        super().__init__()
        
    def after_epoch(self, env, loader=None, loader_valid=None):
        env.checkpoint = True
    

class EarlyStopping(CallbackTemplate):
    '''
    Early stops the training if validation loss doesn't improve after a given patience.
    patience: int       = 
    target: str         = 
    maximize: bool      = 
    skip_epoch: int     =
    '''

    def __init__(self, patience=5, target='valid_metric', maximize=False, skip_epoch=0):
        super().__init__()
        self.state = {
            'patience': patience,
            'target': target,
            'maximize': maximize,
            'skip_epoch': skip_epoch,
            'counter': 0,
            'best_score': None,
            'best_epoch': None
        }

    def after_epoch(self, env, loader=None, loader_valid=None):
        score = env.state[self.state['target']]
        epoch = env.state['epoch'] # local epoch
        if epoch < self.state['skip_epoch'] or epoch == 0:
            self.state['best_score'] = score
            self.state['best_epoch'] = env.global_epoch
            env.checkpoint = True
            env.state['best_score'] = self.state['best_score']
            env.state['best_epoch'] = self.state['best_epoch']
        else:
            if (self.state['maximize'] and score > self.state['best_score']) or \
                    (not self.state['maximize'] and score < self.state['best_score']):
                self.state['best_score'] = score
                self.state['best_epoch'] = env.global_epoch
                self.state['counter'] = 0
                env.checkpoint = True
                env.state['best_score'] = self.state['best_score']
                env.state['best_epoch'] = self.state['best_epoch']
            else:
                self.state['counter'] += 1

            env.state['patience'] = self.state['counter']
            if self.state['counter'] >= self.state['patience']:
                env.stop_train = True

    def state_dict(self):
        return self.state

    def load_state_dict(self, checkpoint):
        self.state = checkpoint

    def __repr__(self):
        return f'EarlyStopping(patience={self.state["patience"]}, skip_epoch={self.state["skip_epoch"]})'


class CollectTopK(CallbackTemplate):
    '''
    Collect top k checkpoints for weight average snapshot
    k: int              = 
    target: str         = 
    maximize: bool      = 
    '''

    def __init__(self, k=3, target='valid_metric', maximize=False, ):
        super().__init__()
        self.state = {
            'k': k,
            'target': target,
            'maximize': maximize,
            'best_scores': np.array([]),
            'best_epochs': np.array([]),
            'counter': 0
        }

    def after_epoch(self, env, loader=None, loader_valid=None):
        score = env.state[self.state['target']]
        epoch = env.state['epoch'] # local epoch
        if len(self.state['best_scores']) < self.state['k']:
            self.state['best_scores'] = np.append(self.state['best_scores'], score)
            self.state['best_epochs'] = np.append(self.state['best_epochs'], env.global_epoch)
            if self.state['maximize']:
                rank = np.argsort(-self.state['best_scores'])
            else:
                rank = np.argsort(self.state['best_scores'])
                
            env.checkpoint = True
            env.state['best_score'] = self.state['best_scores'][rank][0]
            env.state['best_epoch'] = self.state['best_epochs'][rank][0]

        elif (self.state['maximize'] and score > np.min(self.state['best_scores'])) or \
                (not self.state['maximize'] and score < np.max(self.state['best_scores'])):
            if self.state['maximize']:
                del_idx = np.argmin(self.state['best_scores'])
            else:
                del_idx = np.argmax(self.state['best_scores'])
            self.state['best_scores'] = np.delete(self.state['best_scores'], del_idx)
            self.state['best_epochs'] = np.delete(self.state['best_epochs'], del_idx)
            self.state['best_scores'] = np.append(self.state['best_scores'], score)
            self.state['best_epochs'] = np.append(self.state['best_epochs'], env.global_epoch)
            if self.state['maximize']:
                rank = np.argsort(-self.state['best_scores'])
            else:
                rank = np.argsort(self.state['best_scores'])
                
            env.checkpoint = True
            env.state['best_score'] = self.state['best_scores'][rank][0]
            env.state['best_epoch'] = self.state['best_epochs'][rank][0]
            self.state['counter'] = 0
        else:
            self.state['counter'] += 1
        
        env.state['patience'] = self.state['counter']

    def __repr__(self):
        return f'CollectTopK(k={self.state["k"]})'