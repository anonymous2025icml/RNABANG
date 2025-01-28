import os
import pandas as pd
import logging
import random
import numpy as np
from typing import List, Dict, Any
import collections
import sqlite3
import io
import gemmi

from data import rigid_utils as ru

from torch.utils import data
import torch


def read_csv_from_db(file_name, db_name):
    conn = sqlite3.connect(f"file:{db_name}.db?mode=ro", uri=True)
    cursor = conn.cursor()

    # Query the file content
    cursor.execute("SELECT content FROM files WHERE name = ?", (file_name,))
    result = cursor.fetchone()
    conn.close()

    if result:
        # Convert binary content back to a DataFrame
        content = result[0]
        return pd.read_csv(io.BytesIO(content))
    else:
        raise FileNotFoundError(f"File '{file_name}' not found in the database.")




def linear_to_4x4(lin_arr):
    arr_4x4 = []
    for vec in lin_arr:
        t = vec[:3]
        e1 = vec[3:6]
        e2 = vec[6:9]
        e3 = vec[9:12]
        mat = np.hstack((e1.reshape(-1,1),e2.reshape(-1,1),e3.reshape(-1,1),t.reshape(-1,1)))
        mat = np.vstack((mat, np.array([0,0,0,1])))
        arr_4x4.append(mat)
    return np.array(arr_4x4)

mapping = {"DT": "U", "DC": "C", "DG": "G", "DA": "A"}
def replace_values(value):
    return mapping.get(value, value)


def find_ct_token(tokens, i):
    current_position = 0
    
    for index, substring in enumerate(tokens):
        current_position += len(substring)
        
        if current_position > i:
            return index
    
    return None

def middle_element(sorted_array):
    n = len(sorted_array)
    if n % 2 == 1:
        middle_element = sorted_array[n // 2]
    else:
        middle_element = sorted_array[n // 2 - 1]
    return middle_element

def random_subset(array, gap=10):
    
    subsets = []
    current_subset = [array[0]]

    for i in range(1, len(array)):
        if array[i] - array[i - 1] < gap:
            current_subset.append(array[i])
        else:
            subsets.append(current_subset)
            current_subset = [array[i]]

    subsets.append(current_subset)

    return np.array(random.choice(subsets))


def random_middle_element(array, gap=10):
    
    subsets = []
    current_subset = [array[0]]

    for i in range(1, len(array)):
        if array[i] - array[i - 1] < gap:
            current_subset.append(array[i])
        else:
            subsets.append(current_subset)
            current_subset = [array[i]]

    subsets.append(current_subset)

    return middle_element(np.array(random.choice(subsets)))



def pad_features(name, vals, pad_length, pad_ind):

    padded_vals = []
    
    if isinstance(vals[0], torch.Tensor):
        if 'idx' in name:
            for val in vals:
                fill_size = pad_length-val.shape[-1]
                fill_tensor = torch.full((1, fill_size), torch.tensor(0, requires_grad=False).to(val))
                padded_vals.append(torch.cat((val, fill_tensor), dim=-1))
        elif 'type' in name or 'tar' in name:
            for val in vals:
                fill_size = pad_length-val.shape[-1]
                fill_tensor = torch.full((1, fill_size), torch.tensor(pad_ind, requires_grad=False).to(val))
                padded_vals.append(torch.cat((val, fill_tensor), dim=-1))
        elif 'mask' in name:
            for val in vals:
                fill_size = pad_length-val.shape[-1]
                fill_tensor = torch.full((1, fill_size), torch.tensor(0, requires_grad=False).to(val))
                padded_vals.append(torch.cat((val, fill_tensor), dim=-1))
        elif 'ct' in name or 'rna' in name:
            padded_vals = vals
        else:
            raise ValueError(f'Invalid feature name: {name}')

    elif isinstance(vals[0], ru.Rigid):
        for val in vals:
            fill_size = pad_length-val.shape[-1]
            fill_tensor = ru.Rigid.identity((1,fill_size), requires_grad=False)
            padded_vals.append(ru.Rigid.cat((val, fill_tensor), dim=-1))
    else:
        raise ValueError(f'Invalid feature instance: {type(vals[0])}')

    return padded_vals

def cat_features(dicks, pad_na_ind, pad_aa_ind):

    combined_dict = collections.defaultdict(list)
    names=[]
    lengths_na=[]
    lengths_aa=[]

    for chain_dict in dicks:
        for feat_name, feat_val in chain_dict[0].items():
            feat_val = feat_val[None]
            combined_dict[feat_name].append(feat_val)
        
        names.append(chain_dict[1])
        lengths_na.append(chain_dict[0]['ttype_na'].shape[-1])
        lengths_aa.append(chain_dict[0]['rtype_aa'].shape[-1])

    pad_length_na = max(lengths_na)
    pad_length_aa = max(lengths_aa)

    for feat_name, feat_vals in combined_dict.items():

        if 'na' in feat_name:
            pad_feat_vals = pad_features(feat_name, feat_vals, pad_length_na, pad_na_ind)
        elif 'aa' in feat_name:
            pad_feat_vals = pad_features(feat_name, feat_vals, pad_length_aa, pad_aa_ind)
        
        if isinstance(feat_vals[0], torch.Tensor):
            combined_dict[feat_name] = torch.cat(pad_feat_vals, dim=0)
        elif isinstance(feat_vals[0], ru.Rigid):
            combined_dict[feat_name] = ru.Rigid.cat(pad_feat_vals, dim=0)
        else:
            raise ValueError(f'Invalid feature instance: {type(feat_vals[0])}')

    pad_na = []
    for n in lengths_na:
        pad_na.append(torch.cat((torch.ones(n), torch.zeros(pad_length_na - n)))[None])
    combined_dict['pad_na'] = torch.cat(pad_na, dim=0).to(torch.int)

    pad_aa = []
    for n in lengths_aa:
        pad_aa.append(torch.cat((torch.ones(n), torch.zeros(pad_length_aa - n)))[None])
    combined_dict['pad_aa'] = torch.cat(pad_aa, dim=0).to(torch.int)

    return (combined_dict, names)


def create_data_loader(
        torch_dataset: data.Dataset,
        tokenizer,
        batch_size,
        sampler=None,
        num_workers=0,
        drop_last=False,
        prefetch_factor=2):
    """Creates a data loader with jax compatible data structures."""

    pad_na_ind = tokenizer.restypes_order_na.get('<pad>', tokenizer.restypes_num_na)
    pad_aa_ind = tokenizer.restypes_order_aa.get('<pad>', tokenizer.restypes_num_aa)

    persistent_workers = True if num_workers > 0 else False
    prefetch_factor = 2 if num_workers == 0 else prefetch_factor
    return data.DataLoader(
        torch_dataset,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=lambda x: cat_features(x, pad_na_ind, pad_aa_ind),
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=True,
        drop_last=drop_last,
        # Need fork https://github.com/facebookresearch/hydra/issues/964
        multiprocessing_context='fork' if num_workers != 0 else None,
        )



def create_ct_reg_mask(conts, n):

    def f(i,j,ct):

        part1 = (i > ct) & (2*ct-i+2 <= j) & (j <= i)
    
        part2 = (i <= ct) & (2*ct-i+2 >= j) & (j >= i)
        
        return (part1 | part2).int()

    batch_size = conts.shape[0]
    mask = torch.empty(batch_size, n, n)

    i_indices, j_indices = torch.meshgrid(torch.arange(n), torch.arange(n), indexing='ij')

    mask = f(i_indices.repeat(batch_size,1,1).to(conts), j_indices.repeat(batch_size,1,1).to(conts), conts[:,None,None])

    mask.requires_grad=False

    return mask


def rigidFrom3points(x1, x2, x3):
    v1 = x3-x2
    v2 = x1-x2
    e1 = v1/np.linalg.norm(v1)
    u2 = v2-e1*(e1.T@v2)
    e2 = u2/np.linalg.norm(u2)
    e3 = np.cross(e1,e2)
    return(x2,e1,e2,e3)



    