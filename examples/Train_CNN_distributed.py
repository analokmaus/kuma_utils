from dataclasses import dataclass, asdict
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
import timm
from pathlib import Path
import wandb

from kuma_utils.torch import TorchTrainer, TorchLogger
from kuma_utils.torch.callbacks import EarlyStopping, SaveSnapshot
from kuma_utils.torch.hooks import SimpleHook
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
        root='input', train=True, download=True, transform=transform)
    test = torchvision.datasets.CIFAR10(
        root='input', train=False, download=True, transform=transform)
    return train, test


def split_dataset(dataset, index):
    new_dataset = deepcopy(dataset)
    new_dataset.data = new_dataset.data[index]
    new_dataset.targets = np.array(new_dataset.targets)[index]
    return new_dataset


def get_model(num_classes):
    model = timm.create_model('tf_efficientnet_b0_ns', pretrained=True, num_classes=num_classes)
    return model

cfg = Config(
    num_workers=32, 
    batch_size=2048,
    num_epochs=10,
    early_stopping_rounds=5,
)
export_dir = Path('results/demo')
export_dir.mkdir(parents=True, exist_ok=True)

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
    logger = TorchLogger(
        path='results/demo/log', 
        log_items='epoch train_loss train_metric valid_loss valid_metric learning_rate early_stop', 
        file=True,
        use_wandb=True, wandb_params={
            'project': 'kuma_utils_demo', 
            'group': 'demo_cross_validation_ddp',
            'name': f'fold{fold}',
            'config': asdict(cfg),
        })
    
    trn = TorchTrainer(model, serial=f'fold{fold}')
    trn.train(
        loader=loader_train,
        loader_valid=loader_valid,
        criterion=nn.CrossEntropyLoss(),
        eval_metric=Accuracy().torch, 
        monitor_metrics=[
            Accuracy().torch
        ],
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_target='valid_loss', # ReduceLROnPlateau reads metric each epoch
        num_epochs=cfg.num_epochs,
        hook=SimpleHook(evaluate_in_batch=False),
        callbacks=[
            EarlyStopping(
                patience=cfg.early_stopping_rounds, 
                target='valid_metric', 
                maximize=True),
            SaveSnapshot() # Default snapshot path: {export_dir}/{serial}.pt
        ],
        logger=logger, 
        export_dir=export_dir,
        parallel='ddp', # Supported parallel methods: 'dp', 'ddp'
        fp16=True, # Pytorch mixed precision
        deterministic=True, 
        random_state=0, 
        progress_bar=True # Progress bar shows batches done
    )

    oof = trn.predict(loader_valid)
    predictions.append(trn.predict(loader_test))

    score = Accuracy()(valid_fold.targets, oof)
    print(f'Folf{fold} score: {score:.6f}')
