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
    batch_size=2048,
    num_epochs=20,
    early_stopping_rounds=5,
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
        train_fold, batch_size=cfg.batch_size, num_workers=cfg.num_workers, 
        shuffle=True, pin_memory=True)
    loader_valid = D.DataLoader(
        valid_fold, batch_size=cfg.batch_size, num_workers=cfg.num_workers, 
        shuffle=False, pin_memory=True)
    loader_test = D.DataLoader(
        test, batch_size=cfg.batch_size, num_workers=cfg.num_workers, 
        shuffle=False, pin_memory=True)

    model = get_model(num_classes=len(train.classes))
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2)
    
    trn = TorchTrainer(model, serial=f'fold{fold}')
    trn.scheduler_target = 'valid_metric' # ReduceLROnPlateau reads metric each epoch
    trn.progress_bar = True # Progress bar shows batches done
    trn.train(
        loader=loader_train,
        loader_valid=loader_valid,
        loader_test=loader_test,
        criterion=nn.CrossEntropyLoss(),
        eval_metric=Accuracy().torch, 
        monitor_metrics=[
            Accuracy().torch
        ],
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=cfg.num_epochs,
        hook=SimpleHook(evaluate_batch=False),
        callbacks=[
            EarlyStopping(
                patience=cfg.early_stopping_rounds, 
                target='valid_metric', 
                maximize=True)
        ],
        logger=TorchLogger(
            path=f'results/demo/fold{fold}', 
            log_items='epoch train_loss valid_loss valid_metric learning_rate patience', 
            file=True), 
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
classes ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
fold0 starting
train: 40000 / valid: 10000
TorchLogger created at 20/12/10:08:50:15
08:50:15 Mixed precision training on torch amp.
08:50:15 DistributedDataParallel on devices [0, 1, 2, 3]
08:50:15 Model is on cuda
08:50:15 DDP on tcp://127.0.0.1:49051
08:50:23 device[1] initialized.
08:50:27 device[2] initialized.
08:50:31 device[0] initialized.
08:50:31 device[3] initialized.
08:50:37 device[0] ready to train.
08:50:37 device[3] ready to train.
08:50:37 device[2] ready to train.
08:50:37 device[1] ready to train.
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:17<00:00,  1.17it/s]
valid: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  3.70it/s]
08:50:55 Epoch  1/20 | train_loss=1.335771 | valid_loss=7.963142 | valid_metric=0.567400 | learning_rate=[0.002000] | 
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:16<00:00,  1.22it/s]
valid: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  3.48it/s]
08:51:14 Epoch  2/20 | train_loss=0.555175 | valid_loss=0.658680 | valid_metric=0.787800 | learning_rate=[0.002000] | 
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:17<00:00,  1.15it/s]
valid: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  3.58it/s]
08:51:34 Epoch  3/20 | train_loss=0.262158 | valid_loss=0.704435 | valid_metric=0.797100 | learning_rate=[0.002000] | 
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:15<00:00,  1.26it/s]
valid: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  3.57it/s]
...
08:56:11 Epoch 18/20 | train_loss=0.006254 | valid_loss=0.713796 | valid_metric=0.857000 | learning_rate=[0.001000] | 
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:17<00:00,  1.15it/s]
valid: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  3.57it/s]
08:56:31 Epoch 19/20 | train_loss=0.001824 | valid_loss=0.729091 | valid_metric=0.862800 | learning_rate=[0.001000] | 
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:17<00:00,  1.15it/s]
valid: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  3.40it/s]
08:56:51 Epoch 20/20 | train_loss=0.000750 | valid_loss=0.756801 | valid_metric=0.861000 | learning_rate=[0.001000] | (*1)
08:56:58 Best epoch is [19], best score is [0.8628].
inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:04<00:00,  1.18it/s]
inference: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:03<00:00,  1.30it/s]
```

# Module description
## `kuma_utils.torch.EarlyStopping`
```python
# Sample
EarlyStopping(
    patience: int = 5, 
    target: str = 'valid_metric', 
    maximize: bool = False, 
    skip_epoch: int = 0 
)
```
| argument   | description                                                                                         |
|------------|-----------------------------------------------------------------------------------------------------|
| patience   | Epochs to wait before early stop                                                                    |
| target     | Variable name to watch (choose from  `['train_loss', 'train_metric', 'valid_loss', 'valid_metric']`) |
| maximize   | Whether to maximize the target                                                                      |
| skip_epoch | Epochs to skip before early stop counter starts                                                     |


## `kuma_utils.torch.TorchLogger`
```python
TorchLogger(
    path: (str, pathlib.Path),
    log_items: (list, str) = [
        'epoch', 'train_loss', 'valid_loss', 'train_metric', 'valid_metric',
        'train_monitor', 'valid_monitor', 'learning_rate', 'patience'
        ],
    verbose_eval: int = 1,
    stdout: bool = True, 
    file: bool = False
)
```
| argument     | description                                                                                                                                                                                                        |
|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| path         | Path to export log.                                                                                                                                                                                                |
| log_items    | Items to be shown in log. Must be a combination of the following items:  `['epoch',  'train_loss', 'valid_loss', 'train_metric' , 'valid_metric', 'train_monitor',  'valid_monitor', 'learning_rate', 'patience']` |
| verbose_eval | How often (unit: epoch) to log                                                                                                                                                                                     |
| stdout       | Print log to stdout.                                                                                                                                                                                               |
| file         | Write log to the path. (False by default)                                                                                                                                                                          |


## Hook
Hook is used to specify detailed training and evaluation process.
Basically it is not necessary to modify this part, but in some special cases such like

- training a Graph Neural Network which takes multiple arguments in `.forward`
- training with a special metric which requires extra variables (other than predictions and targets)
- calculate metrics on whole dataset (not in each mini-batch)

A Hook class contains the following functions:
```python
class SampleHook:
    def forward_train(self, trainer, inputs):
        target = inputs[-1]
        approx = trainer.model(inputs[0])
        loss = trainer.criterion(approx, target)
        storage['approx'].append(approx)
        storage['target'].append(target)
        return approx, target, loss

    def forward_test(self, trainer, inputs):
        approx = trainer.model(inputs[0])
        return approx

    def evaluate_epoch(self, trainer):
        metric_total = trainer.eval_metric(
                    storage['approx'], storage['target'])
                
        monitor_metrics_total = []
        for monitor_metric in trainer.monitor_metrics:
            monitor_metrics_total.append(
                monitor_metric(storage['approx'], storage['target']))
        return metric_total, monitor_metrics_total
```
`.forward_train()` is called in each mini-batch in training and validation loop.

`.forward_test()` is called in each mini-batch in inference loop.

`.evaluate_epoch()` is called at the end of each training and validation loop.

Note that `trainer.epoch_storage` is a dicationary object you can use freely. 
In `SampleHook`,  predictions and targets are added to storage in each mini-batch, 
and at the end of loop, metrics are calculated on the whole dataset 
(tensors are concatenated automatically).

## Callback
TODO