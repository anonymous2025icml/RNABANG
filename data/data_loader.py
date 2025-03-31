import math
from typing import Optional

import torch
import torch.distributed as dist

import os
import numpy as np
import torch
import pandas as pd
import logging
import random
import sqlite3

from data import utils as du
from data import rigid_utils




class Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            csv,
            data_conf,
            tokenizer,
            is_training
        ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._data_conf = data_conf
        self.csv = csv

        if self._is_training: self.max_len = self._data_conf.max_len

        self.scale_pos = lambda x: x * 0.1
        self.scale_rigids = lambda x: x.apply_trans_fn(self.scale_pos)

        self.tokenizer = tokenizer


    @property
    def is_training(self):
        return self._is_training

    @property
    def data_conf(self):
        return self._data_conf



    def _process_csv_row(self, feats_aa, feats_na=None, conts_na=None):
        
        ridx_aa = feats_aa['id'].values
        ridx_aa = ridx_aa - np.min(ridx_aa)

        rtype_aa = feats_aa['residue'].values.tolist()
        rtype_aa = self.tokenizer.tokenize_aa(rtype_aa)

        linears_aa = feats_aa.iloc[:,2:14].values
        com = np.sum(linears_aa, axis=0)[:3]/len(linears_aa)
        linears_aa[:,:3] -= com

        frames_aa = du.linear_to_4x4(linears_aa)
        T_aa = torch.tensor(frames_aa).float()
        T_aa = rigid_utils.Rigid.from_tensor_4x4(T_aa.unsqueeze(1))[:, 0]
        T_aa = self.scale_rigids(T_aa)

        if feats_na is not None:
            
            vectorized_replace = np.vectorize(du.replace_values)
            rtype_na = vectorized_replace(feats_na['residue'].values)

            ttype_na = rtype_na.copy()
            tidx_na = feats_na['id'].values

            cont_tokens = np.array([cont_idx - np.min(tidx_na)  for cont_idx in conts_na])
            tidx_na = tidx_na - np.min(tidx_na) + 1
            cont_token = np.random.choice(cont_tokens)
            

            ttar_na = np.concatenate([['<eos>'], ttype_na, ['<eos>']])
            ttype_na = np.concatenate([['<eos>'], ttype_na[:cont_token + 1], ['<cntl>', '<cntr>'], ttype_na[cont_token + 1:], ['<eos>']])
            cont_token += 1

            tidx_na = np.concatenate([[0], tidx_na[:cont_token], [cont_token, cont_token + 1], tidx_na[cont_token:], [tidx_na[-1] + 1]])

            if len(ttype_na) > self.max_len+4:

                if cont_token-self.max_len//2 <= 0:

                    tidx_na = tidx_na[:self.max_len+4]
                    ttype_na = ttype_na[:self.max_len+4]
                    ttar_na = ttar_na[:self.max_len+2]

                elif cont_token+self.max_len//2+4 >= len(ttype_na)-1:

                    cont_token -= len(ttype_na)-self.max_len-4
                    
                    tidx_na = tidx_na[-self.max_len-4:]
                    ttype_na = ttype_na[-self.max_len-4:]
                    ttar_na = ttar_na[-self.max_len-2:]
                    
                else:

                    tidx_na = tidx_na[cont_token-self.max_len//2:cont_token+self.max_len//2+4]
                    ttype_na = ttype_na[cont_token-self.max_len//2:cont_token+self.max_len//2+4]
                    ttar_na = ttar_na[cont_token-self.max_len//2:cont_token+self.max_len//2+2]

                    cont_token = self.max_len//2

            ttar_na = np.concatenate([ttar_na, ['<pad>', '<pad>']])

            ttar_na = self.tokenizer.encode_na(ttar_na)

        else:

            ttype_na = ['<cntl>', '<cntr>']
            tidx_na = [0, 1]
            cont_token = -1

        ttype_na = self.tokenizer.encode_na(ttype_na)

        final_feats = {
            'ttype_na': torch.tensor(ttype_na).to(torch.int),
            'rtype_aa': torch.tensor(rtype_aa).to(torch.int),
            'tidx_na': torch.tensor(tidx_na).to(torch.int16),
            'ridx_aa': torch.tensor(ridx_aa).to(torch.int16),
            'T_aa': T_aa,
            'ct_na': torch.tensor(cont_token+1).to(torch.int16)
        }

        if feats_na is not None:
            final_feats['ttar_na'] = torch.tensor(ttar_na).to(torch.int)
        
        
        return final_feats


    def __len__(self):
        return len(self.csv)

    def __getitem__(self, example_idx):

        # Sample data example.
        csv_row = self.csv.iloc[example_idx]
        id = csv_row['id']
        
        feats_aa = du.read_csv_from_db(f'{csv_row.id_aa}.csv', f'{self.data_conf.dataset_path}/dataset_aa')
        rna = csv_row.rna

        if self.is_training:
            feats_na = du.read_csv_from_db(f'{csv_row.id_na}.csv', f'{self.data_conf.dataset_path}/dataset_na')
            conts_na = sorted([int(elt) for elt in csv_row.ct_na.split(',')])
            complex_feats = self._process_csv_row(feats_aa, feats_na, conts_na)
        else:
            complex_feats = self._process_csv_row(feats_aa)
        
        complex_feats['rna'] = torch.tensor(rna).to(torch.int)

        return complex_feats, id


class Sampler(torch.utils.data.Sampler):

    def __init__(
            self,
            *,
            data_conf,
            dataset,
            sample_mode,
            sample_num
        ):
        
        self._data_conf = data_conf
        self._dataset = dataset
        self._data_csv = self._dataset.csv
        self._data_csv['index'] = list(range(len(self._data_csv)))
        self._sample_mode = sample_mode
        self._sample_num = sample_num

        self.epoch = 0

        if self._sample_mode == 'cluster_batch':
            self.sampler_len = self._sample_num*len(set(self._data_csv['clust'].values))
        elif self._sample_mode == 'simple_batch':
            self.sampler_len = len(self._data_csv)
        else:
            raise ValueError(f'Invalid sample mode: {self._sample_mode}')


    def __iter__(self):

        if self._sample_mode == 'cluster_batch':
            sampled_clusters = self._data_csv.groupby('clust').sample(self._sample_num, random_state=self.epoch, replace=True)
            self._dataset_indices = sampled_clusters.index.tolist()
        elif self._sample_mode == 'simple_batch':
            self._dataset_indices = self._data_csv['index'].tolist()
        else:
            raise ValueError(f'Invalid sample mode: {self._sample_mode}')

        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(len(self._dataset_indices), generator=g).tolist()
        self._dataset_indices = [self._dataset_indices[i] for i in indices]

        return iter(self._dataset_indices)
        
    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.sampler_len



class DistributedSampler(torch.utils.data.Sampler):

    def __init__(self, 
                *,
                data_conf,
                dataset,
                sample_mode,
                sample_num,
                num_replicas,
                rank
                ):

        self._data_conf = data_conf
        self._dataset = dataset
        self._data_csv = self._dataset.csv
        self._data_csv['index'] = list(range(len(self._data_csv)))
        self._sample_mode = sample_mode
        self._sample_num = sample_num

        self.epoch = 0

        self.num_replicas = num_replicas
        self.rank = rank

        if self._sample_mode == 'cluster_batch':
            self.sampler_len = self._sample_num*len(set(self._data_csv['clust'].values))
        elif self._sample_mode == 'simple_batch':
            self.sampler_len = len(self._data_csv)
        else:
            raise ValueError(f'Invalid sample mode: {self._sample_mode}')

        self.sampler_len = math.ceil(self.sampler_len / self.num_replicas)
        self.total_size = self.sampler_len * self.num_replicas
        

    def __iter__(self) :

        if self._sample_mode == 'cluster_batch':
            sampled_clusters = self._data_csv.groupby('clust').sample(self._sample_num, random_state=self.epoch, replace=True)
            self._dataset_indices = sampled_clusters.index.tolist()
        elif self._sample_mode == 'simple_batch':
            self._dataset_indices = self._data_csv['index'].tolist()
        else:
            raise ValueError(f'Invalid sample mode: {self._sample_mode}')
        
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(len(self._dataset_indices), generator=g).tolist()
        self._dataset_indices = [self._dataset_indices[i] for i in indices]

        padding_size = self.total_size - len(self._dataset_indices)
        self._dataset_indices = self._dataset_indices + self._dataset_indices[:padding_size]
        
        assert len(self._dataset_indices) == self.total_size

        indices = self._dataset_indices[self.rank:self.total_size:self.num_replicas]
        
        assert len(indices) == self.sampler_len

        return iter(indices)

    def __len__(self) -> int:
        return self.sampler_len

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
    


    

class TestSampler(torch.utils.data.Sampler):

    def __init__(
            self,
            *,
            data_conf,
            dataset,
            nof_samples,
        ):
        
        self._data_conf = data_conf
        self._dataset = dataset
        self._data_csv = self._dataset.csv
        self._dataset_indices = list(range(len(self._data_csv)))
        self._data_csv['index'] = self._dataset_indices
        self._nof_samples = nof_samples
        self.epoch = 0
        self.sampler_len = nof_samples

        
    
    def __iter__(self):
        
        if self._nof_samples == 'all':
            indices = self._dataset_indices
        else:
            random.shuffle(self._dataset_indices)
            if self._nof_samples > len(self._dataset_indices):
                indices = random.choices(self._dataset_indices, self._nof_samples)
            else:
                indices = random.sample(self._dataset_indices, self._nof_samples)
        return iter(indices)
        
        

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.sampler_len