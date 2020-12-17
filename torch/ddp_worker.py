import pickle
import argparse
from pathlib import Path
import sys
sys.path.append('.')
import kuma_utils


def ddp_worker(path, rank):
    with open(path, 'rb') as f:
        ddp_tmp = pickle.load(f)
    trainer = ddp_tmp['trainer']
    dist_url = ddp_tmp['dist_url']
    loader =  ddp_tmp['loader']
    loader_valid = ddp_tmp['loader_valid']
    num_epochs = ddp_tmp['num_epochs']
    
    assert rank < trainer.world_size
    trainer._train_ddp(rank, dist_url, loader, loader_valid, num_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--path', type=str, required=True)
    opt = parser.parse_args()
    ddp_worker(opt.path, opt.rank)
