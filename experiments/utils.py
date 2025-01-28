import pickle
import os
import copy
from typing import Any
from typing import List, Dict, Any
import collections
import random
import pandas as pd
import math

from torch.optim.lr_scheduler import LRScheduler

import io
import torch
import torch.distributed as dist


def get_ddp_info():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    node_id = rank // world_size
    return {"node_id": node_id, "local_rank": local_rank, "rank": rank, "world_size": world_size}


def create_split(conf, _log):

    if conf.experiment.load_from_ckpt:
        ckpt_dir = conf.experiment.load_from_ckpt
        ckpt_files = [x for x in os.listdir(ckpt_dir) if '.csv' in x]

        if len(ckpt_files) != 1:
            raise ValueError(f'Ambiguous validation set in {ckpt_dir}')
        
        ckpt_name = ckpt_files[0]
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        valid_csv = pd.read_csv(ckpt_path)#.drop(columns = ['Unnamed: 0'])

        pdb_csv = pd.read_csv(conf.data.csv_path)
        train_csv = pd.merge(pdb_csv, valid_csv, how='outer', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1).drop_duplicates().reset_index()
        train_csv = train_csv.reset_index(drop=True)

        train_clusters = list(set(train_csv['clust'].values))

    if conf.data.valid_csv_path:

        valid_csv = pd.read_csv(conf.data.valid_csv_path)#.drop(columns = ['Unnamed: 0'])

        pdb_csv = pd.read_csv(conf.data.csv_path)
        train_csv = pd.merge(pdb_csv, valid_csv, how='outer', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1).drop_duplicates().reset_index()
        train_csv = train_csv.reset_index(drop=True)

        train_clusters = list(set(train_csv['clust'].values))
        
    else:
        pdb_csv = pd.read_csv(conf.data.csv_path)
        train_idx = []
        clusters = list(set(pdb_csv['clust_aa'].values))
        
        random.seed(33)
        random.shuffle(clusters)

        train_clusters = clusters[:int(conf.data.split*len(clusters))]

        for cluster in train_clusters:
            train_idx += list(pdb_csv[pdb_csv['clust_aa'] == cluster].index)

        valid_idx = list(set(pdb_csv.index).difference(train_idx))

        train_csv = pdb_csv.loc[train_idx]
        valid_csv = pdb_csv.loc[valid_idx]

        train_csv = train_csv.reset_index(drop=True)
        valid_csv = valid_csv.reset_index(drop=True)

        train_clusters = list(set(train_csv['clust'].values))


    _log.info(f'Training: {len(train_csv)} examples, {len(train_clusters)} clusters')
    _log.info(f'Validating: {len(valid_csv)} examples')
    
    return train_csv, valid_csv


move_to_np = lambda x: x.cpu().detach().numpy()


def divide_dict(log_dict):
    output_dict = {}
    for key in log_dict:
        if "_loss" in key:
            name = key.split('_')[0]
            sample_key = f"{name}_samples"
            if sample_key in log_dict: 
                output_dict[key] = move_to_np(log_dict[key] / log_dict[sample_key])
    return output_dict


def flatten_dict(raw_dict):
    """Flattens a nested dict."""
    flattened = []
    for k, v in raw_dict.items():
        if isinstance(v, dict):
            flattened.extend([
                (f'{k}:{i}', j) for i, j in flatten_dict(v)
            ])
        else:
            flattened.append((k, v))
    return flattened


class CPU_Unpickler(pickle.Unpickler):
    """Pytorch pickle loading workaround.

    https://github.com/pytorch/pytorch/issues/16797
    """
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def write_pkl(save_path: str, pkl_data: Any, create_dir: bool = False, use_torch=False):
    """Serialize data into a pickle file."""
    if create_dir:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if use_torch:
        torch.save(pkl_data, save_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(save_path, 'wb') as handle:
            pickle.dump(pkl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pkl(read_path: str, verbose=True, use_torch=False, map_location=None):
    """Read data from a pickle file."""
    try:
        if use_torch:
            return torch.load(read_path, map_location=map_location, weights_only=False)
        else:
            with open(read_path, 'rb') as handle:
                return pickle.load(handle)
    except Exception as e:
        try:
            with open(read_path, 'rb') as handle:
                return CPU_Unpickler(handle).load()
        except Exception as e2:
            if verbose:
                print(f'Failed to read {read_path}. First error: {e}\n Second error: {e2}')
            raise(e)



def write_checkpoint(
        ckpt_path: str,
        model,
        conf,
        optimizer,
        epoch,
        step,
        logger=None,
        use_torch=True,
    ):
    """Serialize experiment state and stats to a pickle file.

    Args:
        ckpt_path: Path to save checkpoint.
        conf: Experiment configuration.
        optimizer: Optimizer state dict.
        epoch: Training epoch at time of checkpoint.
        step: Training steps at time of checkpoint.
        exp_state: Experiment state to be written to pickle.
        preds: Model predictions to be written as part of checkpoint.
    """
    ckpt_dir = os.path.dirname(ckpt_path)
    '''for fname in os.listdir(ckpt_dir):
        if '.pkl' in fname or '.pth' in fname:
            os.remove(os.path.join(ckpt_dir, fname))'''
    if logger is not None:
        logger.info(f'Serializing experiment state to {ckpt_path}')
    else:
        print(f'Serializing experiment state to {ckpt_path}')
    write_pkl(
        ckpt_path,
        {
            'model': model,
            'conf': conf,
            'optimizer': optimizer,
            'epoch': epoch,
            'step': step
        },
        use_torch=use_torch)
    







def get_lr_lambda_gamma(step, warmup_steps=1000, gamma=0.99, decay_interval=1000):
    if step < warmup_steps:
        # Linear warm-up phase
        return float(step) / float(max(1, warmup_steps))
    else:
        # After warm-up, decay the learning rate
        return gamma ** ((step - warmup_steps) // decay_interval)

def get_lr_lambda_cos(step, warmup_steps=1000, period=10000):
    if step < warmup_steps:
        # Linear warm-up phase
        return float(step) / float(max(1, warmup_steps))
    else:
        # After warm-up, decay the learning rate
        return 0.5*(1.0+math.cos(float(step-warmup_steps)/float(period)*math.pi))





class CosineLRScheduler(LRScheduler):
    def __init__(self, optimizer, initial_lr, warmup_steps=1000, period=10000, last_epoch=-1):
        """
        Custom learning rate scheduler inheriting from PyTorch's LRScheduler.
        
        Args:
            optimizer: The optimizer whose learning rate needs scheduling.
            initial_lr: The starting learning rate after warmup.
            lr_min: The minimum learning rate for cosine decay.
            last_epoch: The index of the last epoch. Default: -1 (starts from step 0).
        """
        self.initial_lr = initial_lr
        self.lr_min = 0.01*self.initial_lr
        self.warmup_steps = warmup_steps
        self.period = period
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute the learning rate for the current step."""
        step = self._step_count
        if step < self.warmup_steps:
            # Linear warmup phase
            return [self.initial_lr * (step / self.warmup_steps) for _ in self.optimizer.param_groups]
        else:
            # Cosine decay phase
            return [
                self.lr_min + (self.initial_lr - self.lr_min) *
                0.5 * (1 + math.cos((step - self.warmup_steps) / self.period * math.pi))
                for _ in self.optimizer.param_groups
            ]




def soften_mask(mask, t, N=10000):

    if t > N:
        return mask
    else:
        scale_factor = 1 + 9 * (N - t) / N

        scaled_mask = mask * scale_factor

        # Cap the values at 1.0
        softened_mask = torch.clamp(scaled_mask, max=1.0)
    
    return softened_mask