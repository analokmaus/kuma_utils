from .base import CallbackTemplate
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel


class BestEpoch(CallbackTemplate):
    '''
    Path priority: path argument > BestEpoch.path > trainer.snapshot_path
    '''

    def __init__(self, path=None):
        super().__init__()
        self.path = path

    def save_snapshot(self, trainer, path):
        if isinstance(
                trainer.model,
                (DataParallel, DistributedDataParallel)):
            module = trainer.model.module
        else:
            module = self.model
        
        serialized = {
            'global_epoch': trainer.global_epoch,
            'model': module.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
            'scheduler': trainer.scheduler.state_dict(),
            'state': trainer.state,
            'all_states': trainer._states
        }

        if path is None:
            path = self.path if self.path is not None else trainer.snapshot_path

        if trainer.xla:
            import torch_xla.utils.serialization as xser
            xser.save(serialized, str(path))
        else:
            torch.save(serialized, str(path))

    def load_snapshot(self, trainer, path=None, device=None):
        if path is None:
            path = self.path if self.path is not None else trainer.snapshot_path
        if device is None:
            device = trainer.device
        
        if trainer.xla:
            import torch_xla.utils.serialization as xser
            checkpoint = xser.load(str(path))
        else:
            checkpoint = torch.load(str(path), map_location=device)

        if isinstance(
                trainer.model,
                (DataParallel, DistributedDataParallel)):
            trainer.model.module.load_state_dict(checkpoint['model'])
        else:
            trainer.model.load_state_dict(checkpoint['model'])

        if hasattr(trainer, 'optimizer'):
            trainer.optimizer.load_state_dict(checkpoint['optimizer'])
        if hasattr(trainer, 'scheduler'):
            trainer.scheduler.load_state_dict(checkpoint['scheduler'])
        if hasattr(trainer, 'global_epoch'):
            trainer.global_epoch = checkpoint['global_epoch']
        trainer.state = checkpoint['state']
        trainer._states = checkpoint['all_states']
