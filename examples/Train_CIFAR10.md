# Train CIFAR-10
## Environment
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.87.01    Driver Version: 418.87.01    CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  On   | 00000000:00:1B.0 Off |                    0 |
| N/A   38C    P0    65W / 300W |   2302MiB / 16130MiB |    100%      Default |
+-------------------------------+----------------------+----------------------+
|   1  Tesla V100-SXM2...  On   | 00000000:00:1C.0 Off |                    0 |
| N/A   37C    P0    63W / 300W |   2370MiB / 16130MiB |    100%      Default |
+-------------------------------+----------------------+----------------------+
|   2  Tesla V100-SXM2...  On   | 00000000:00:1D.0 Off |                    0 |
| N/A   40C    P0    66W / 300W |   2370MiB / 16130MiB |    100%      Default |
+-------------------------------+----------------------+----------------------+
|   3  Tesla V100-SXM2...  On   | 00000000:00:1E.0 Off |                    0 |
| N/A   41C    P0    66W / 300W |   2382MiB / 16130MiB |    100%      Default |
+-------------------------------+----------------------+----------------------+
```

## Code
```python
from dataclasses import dataclass
from copy import deepcopy
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
import torch.optim as optim

from kuma_utils.torch import TorchTrainer, TorchLogger, EarlyStopping
from kuma_utils.torch.hooks import SimpleHook, ArgumentSpecifiedHook
from kuma_utils.torch.model_zoo import se_resnext50_32x4d
from kuma_utils.metrics import Accuracy


@dataclass
class Config:
    num_workers: int = 32
    batch_size: int = 64
    num_epochs: int = 100
    early_stopping_rounds: int = 5


def get_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train = torchvision.datasets.CIFAR10(
        root='input', train=True, download=False, transform=transform)
    test = torchvision.datasets.CIFAR10(
        root='input', train=False, download=False, transform=transform)
    return train, test


def split_dataset(dataset, index):
    new_dataset = deepcopy(dataset)
    new_dataset.data = new_dataset.data[index]
    new_dataset.targets = np.array(new_dataset.targets)[index]
    return new_dataset


def get_model(num_classes):
    model = se_resnext50_32x4d(pretrained='imagenet')
    in_features = model.last_linear.in_features
    model.last_linear = nn.Linear(in_features, num_classes)
    return model

cfg = Config(
    num_workers=32, 
    batch_size=256,
    num_epochs=10,
    early_stopping_rounds=2
)

train, test = get_dataset()
print('classes', train.classes)

predictions = []
splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for fold, (train_idx, valid_idx) in enumerate(
    splitter.split(train.targets, train.targets)):

    print(f'fold{fold} starting')

    valid_fold = split_dataset(train, valid_idx)
    train_fold = split_dataset(train, train_idx)

    print(f'train: {len(train_fold)} / valid: {len(valid_fold)}')

    loader_train = D.DataLoader(
        train_fold, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True)
    loader_valid = D.DataLoader(
        valid_fold, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True)
    loader_test = D.DataLoader(
        test, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True)

    model = get_model(num_classes=len(train.classes))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5)
    
    trn = TorchTrainer(model, serial=f'fold{fold}')
    trn.scheduler_target = 'valid_metric' # ReduceLROnPlateau reads metric each epoch
    trn.progress_bar = True # Progress bar shows batches done
    trn.train(
        loader=loader_train,
        loader_valid=loader_valid,
        loader_test=loader_test,
        criterion=nn.CrossEntropyLoss(),
        eval_metric=Accuracy().torch, 
        monitor_metrics=[ # Optional
            Accuracy().torch
        ],
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=cfg.num_epochs,
        hook=SimpleHook(evaluate_batch=False), # Optional
        callbacks=[ # Optional
            EarlyStopping(patience=cfg.early_stopping_rounds, maximize=True)
        ],
        logger=TorchLogger( # Optional
            path='results/demo/log', 
            log_items=[
                'epoch', 'train_loss', 'valid_loss', 'valid_metric', 'valid_monitor', 
                'learning_rate', 'patience'], 
            file=True), # A log file will be export if file=True
        export_dir='results/demo',
        parallel='ddp',
        fp16=True
    )

    oof = trn.outoffold
    predictions.append(trn.prediction)

    break
```
## Results
```
train: 40000 / valid: 10000
TorchLogger created at 20/12/09:09:25:23
09:25:23 Mixed precision training on torch amp.
09:25:23 DistributedDataParallel on devices [0, 1, 2, 3]
09:25:23 Model is on cuda
09:25:23 DDP: tcp://127.0.0.1:56495
09:25:31 device[1] initialized.
09:25:35 device[2] initialized.
09:25:39 device[3] initialized.
09:25:39 device[0] initialized.
09:25:43 device[0] ready to train.
09:25:43 device[3] ready to train.
09:25:43 device[2] ready to train.
09:25:43 device[1] ready to train.
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:09<00:00,  4.13it/s]
valid: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 13.63it/s]
09:25:53 Epoch 1/5 | train_loss=0.308403 | valid_loss=0.188509 | valid_metric=0.749500 | valid_monitor=[0.749500] | learning_rate=[0.001000] | 
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:09<00:00,  4.38it/s]
valid: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 13.63it/s]
09:26:04 Epoch 2/5 | train_loss=0.116023 | valid_loss=0.144912 | valid_metric=0.807900 | valid_monitor=[0.807900] | learning_rate=[0.001000] | 
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:08<00:00,  4.46it/s]
valid: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 13.73it/s]
09:26:15 Epoch 3/5 | train_loss=0.041850 | valid_loss=0.210333 | valid_metric=0.793800 | valid_monitor=[0.793800] | learning_rate=[0.001000] | (*1)
09:26:15 Training stopped by overfit detector.
09:26:20 Best epoch is [2], best score is [0.8079].
```