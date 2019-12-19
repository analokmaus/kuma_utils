import torch
from pathlib import Path
import os


def get_latest_sanpshot(dir, keyword=None):
    if isinstance(dir, str):
        dir = Path(dir)
    if keyword is None:
        files = list(dir.glob('*.pt'))
    else:
        files = list(dir.glob(f'*{keyword}*.pt'))
    if len(files) == 0:
        print('no snapshot found.')
        return None
    file_updates = {file_path: os.stat(
        file_path).st_mtime for file_path in files}
    latest_file_path = max(file_updates, key=file_updates.get)
    print(f'latest snapshot is {str(latest_file_path)}')
    return str(latest_file_path)


def save_snapshots(epoch, model, optimizer, scheduler, path):
    if isinstance(model, torch.nn.DataParallel):
        module = model.module
    else:
        module = model

    torch.save({
        'epoch': epoch + 1,
        'model': module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': None if scheduler.__class__.__name__ == 'CyclicLR' else scheduler.state_dict()
    }, path)


def load_snapshots_to_model(path, model=None, optimizer=None, scheduler=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device)
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])


def load_pretrained(path, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pretrained_weights = torch.load(path, map_location=device)['model']
    model_weights = model.state_dict()
    for layer in model_weights.keys():
        if layer in pretrained_weights.keys():
            model_weights[layer] = pretrained_weights[layer]
    model.load_state_dict(model_weights)


def load_epoch(path):
    checkpoint = torch.load(path)
    return checkpoint['epoch']
