import os
import time
import tree
import numpy as np
import hydra
import torch
import logging
import pandas as pd
from datetime import datetime
import math

from model import main_network
from data import utils as du
from experiments import utils as eu
from data import data_loader
from data import tokenizer
from typing import Dict

from omegaconf import DictConfig, OmegaConf






class Inference:

    def __init__(
            self,
            conf: DictConfig,
        ):

        self._log = logging.getLogger(__name__)

        # Remove static type checking.
        OmegaConf.set_struct(conf, False)

        # Prepare configs.
        self._conf = conf
        self._inf_conf = conf.inference
        self._data_conf = conf.data
        self._tok_conf = conf.tokenizer

        self.sample_mode = conf.inference.sample_mode
        self.max_len = conf.inference.max_len
        self.kp = conf.inference.kp
        self.nof_samples = conf.inference.nof_samples

        # Set-up directories
        self._weights_path = self._inf_conf.model_path
        self._test_set_path = self._inf_conf.testset_path

        output_dir = self._inf_conf.output_dir
        if self._inf_conf.name is None:
            name_string = self._inf_conf.model_path.split('/')[-3]
        else:
            name_string = self._inf_conf.name

        dt_string = datetime.now().strftime("%dD_%mM_%YY_%Hh_%Mm_%Ss")
        self._output_dir = os.path.join(output_dir, name_string, dt_string)
        os.makedirs(self._output_dir, exist_ok=True)
        self._log.info(f'Saving results to {self._output_dir}')

        # GPU mode
        if torch.cuda.is_available():
            self.device = f"cuda:0"
        else:
            self.device = 'cpu'
        self._log.info(f"Using device: {self.device}")

        # Load models and experiment
        self.test_csv = pd.read_csv(self._test_set_path)
        self._load_ckpt()

        config_path = os.path.join(self._output_dir, 'inference_conf.yaml')
        with open(config_path, 'w') as f:
            OmegaConf.save(config=self._conf, f=f)
        self._log.info(f'Saving inference config to {config_path}')
        
       
        

    def _load_ckpt(self):
        """Loads in model checkpoint."""
        self._log.info(f'Loading weights from {self._weights_path}')

        # Read checkpoint
        weights_pkl = eu.read_pkl(self._weights_path, use_torch=True, map_location=self.device)

        weights_pkl['model'] = {key.replace('module.', ''): value for key, value in weights_pkl['model'].items()}

        # Merge base experiment config with checkpoint config.
        self._conf.model = weights_pkl['conf'].model
        
        # Prepare model
        self._model_conf = self._conf.model
        self._tok_conf.vocab_size = weights_pkl['conf'].tokenizer.vocab_size

        # Initialize experiment objects
        self.tokenizer = tokenizer.Tokenizer(self._tok_conf)
        self._model = main_network.MainNetwork(self._model_conf, self.tokenizer)
        
        num_parameters = sum(p.numel() for p in self._model.parameters())
        self._log.info(f'Number of model parameters {num_parameters}')

        # Remove module prefix if it exists.
        model_weights = weights_pkl['model']
        model_weights = {k.replace('module.', ''):v for k,v in model_weights.items()}
        self._model.load_state_dict(model_weights)
        self._model = self._model.to(self.device)
        self._model.eval()
        for m in self._model.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()

    

    @property
    def model(self):
        return self._model

    @property
    def conf(self):
        return self._conf
        

    def run_inference(self):

        test_loader = self.create_test_loader()

        _ = self.inference_fn(test_loader)




    def create_test_loader(self):
        
        test_dataset = data_loader.Dataset(
            self.test_csv,
            self.conf.data,
            self.tokenizer,
            is_training=False
        )

        test_sampler = data_loader.TestSampler(
                data_conf=self.conf.data,
                dataset=test_dataset,
                nof_samples='all',
        )

        test_loader = du.create_data_loader(
            test_dataset,
            self.tokenizer,
            sampler=test_sampler,
            batch_size=1,
            num_workers=self._data_conf.num_loader_workers
        )

        return test_loader
    

    def inference_fn(self, test_loader):


        for test_feats, sample_id in test_loader:

            test_feats = tree.map_structure(lambda x: x.to(self.device), test_feats)

            with torch.no_grad():
                aa_repr = self.model.forward_aa(test_feats)

            for idx in range(self.nof_samples):
                copy_test_feats = tree.map_structure(lambda x: x.clone(), test_feats)
                sampled_output = self.sample(copy_test_feats, aa_repr)

                sampled_seq = sampled_output['seq_na']
                sampled_prob = np.mean(sampled_output['probs_na'][1:-1])

                seq_na_str_uncut = '|'.join(sampled_seq)
                seq_na_str = ''.join([val for val in sampled_seq if val != '<cntl>' and val != '<cntr>' and val != '<eos>'])

                if not test_feats['rna'][0]:
                    seq_na_str = seq_na_str.replace('U', 'T')

                with open(os.path.join(self._output_dir, 'seq_na_uncut.fasta'), 'a') as f:
                    f.write(f'>{sample_id[0]}_{idx},{sampled_prob}\n')
                    f.write(f'{seq_na_str_uncut}\n')

                with open(os.path.join(self._output_dir, 'seq_na.fasta'), 'a') as f:
                    f.write(f'>{sample_id[0]}_{idx},{sampled_prob}\n')
                    f.write(f'{seq_na_str}\n')
            
            self._log.info(f'Done sample {sample_id[0]}')
            
        return 0
    

    def update_batch_l(self, batch, new_token_ids):

        if batch['ttype_na'][0][0] != self.eos_id:

            if new_token_ids[0][0] != self.eos_id:
                self.len_l += len(self.tokenizer.restypes_na[new_token_ids[0][0].tolist()])

            if batch['ttype_na'][0][0] != self.cntl_id:
                batch['tidx_na'] += 1
                batch['tidx_na'] = torch.cat((batch['tidx_na'][...,:1]-1, batch['tidx_na']), dim=-1)
            else:
                batch['tidx_na'] = torch.cat((batch['tidx_na'][...,:1], batch['tidx_na']), dim=-1)

            if self.len_l+self.len_r <= self.max_len:
                batch['ttype_na'] = torch.cat((new_token_ids[:,:1], batch['ttype_na']), dim=-1)
            else:
                eos_to_append = self.eos_id.repeat(batch['ttype_na'].shape[:-1]).to(batch['ttype_na'])
                batch['ttype_na'] = torch.cat((eos_to_append.unsqueeze(-1), batch['ttype_na']), dim=-1)

            batch['ct_na'] += 1

            batch['pad_na'] = torch.cat((batch['pad_na'][...,:1], batch['pad_na']), dim=-1)
        
        return batch

    
    def update_batch_r(self, batch, new_token_ids):

        if batch['ttype_na'][0][-1] != self.eos_id:

            if new_token_ids[0][-1] != self.eos_id:
                self.len_r += len(self.tokenizer.restypes_na[new_token_ids[0][-1].tolist()])

            if batch['ttype_na'][0][-1] != self.cntr_id:
                batch['tidx_na'] = torch.cat((batch['tidx_na'], batch['tidx_na'][...,-1:]+1), dim=-1)
            else:
                batch['tidx_na'] = torch.cat((batch['tidx_na'], batch['tidx_na'][...,-1:]), dim=-1)

            if self.len_r+self.len_l <= self.max_len:
                batch['ttype_na'] = torch.cat((batch['ttype_na'], new_token_ids[:,-1:]), dim=-1)
            else:
                eos_to_append = self.eos_id.repeat(batch['ttype_na'].shape[:-1]).to(batch['ttype_na'])
                batch['ttype_na'] = torch.cat((batch['ttype_na'], eos_to_append.unsqueeze(-1)), dim=-1)
                
            batch['pad_na'] = torch.cat((batch['pad_na'][...,:1], batch['pad_na']), dim=-1)

        return batch
    

    def sample_tokens(self, probs):
        
        if self.sample_mode == 'greedy':
            tokens_ids = torch.argmax(probs, dim=-1)

        elif self.sample_mode == 'topk':
            top_k_probs, top_k_indices = torch.topk(probs, self.kp)
            dist = torch.distributions.Categorical(top_k_probs)
            selected_index = dist.sample()
            tokens_ids = torch.gather(top_k_indices, dim=2, index=selected_index.unsqueeze(-1)).squeeze(-1)

        elif self.sample_mode == 'topp':
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            top_p_mask = cumulative_probs <= self.kp
            top_p_mask[..., 0] = 1
            filtered_probs = sorted_probs * top_p_mask.float()
            filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

            dist=torch.distributions.Categorical(filtered_probs)
            selected_index = dist.sample()
            tokens_ids = torch.gather(sorted_indices, dim=2, index=selected_index.unsqueeze(-1)).squeeze(-1)

        else:
            raise ValueError(f'Invalid sample mode: {self.sample_mode}')

        return tokens_ids
    

    def sample(self, batch, aa_repr):

        self.len_r = 0
        self.len_l = 0
        self.eos_id = torch.tensor(self.tokenizer.tokenize_na(['<eos>'])).to(batch['ttype_na'])
        self.cntl_id = torch.tensor(self.tokenizer.tokenize_na(['<cntl>'])).to(batch['ttype_na'])
        self.cntr_id = torch.tensor(self.tokenizer.tokenize_na(['<cntr>'])).to(batch['ttype_na'])


        while not (batch['ttype_na'][0][0] == self.eos_id and batch['ttype_na'][0][-1] == self.eos_id):

            if batch['ttype_na'][0][-1] != self.eos_id:

                with torch.no_grad():
                    model_out = self.model.forward_na(aa_repr, batch)

                probs = torch.exp(model_out['log_probs'])
                tokens_ids = self.sample_tokens(probs)
                batch = self.update_batch_r(batch, tokens_ids)

            if batch['ttype_na'][0][0] != self.eos_id:

                with torch.no_grad():
                    model_out = self.model.forward_na(aa_repr, batch)

                probs = torch.exp(model_out['log_probs'])
                tokens_ids = self.sample_tokens(probs)
                batch = self.update_batch_l(batch, tokens_ids)

        seq_na = [self.tokenizer.restypes_na[i] for i in batch['ttype_na'][0].tolist()]

        with torch.no_grad():
            model_out = self.model.forward_na(aa_repr, batch)
        probs = torch.exp(model_out['log_probs'])
        probs = probs[0][1:-1].tolist()
        toks = [id for id in batch['ttype_na'][0].tolist() if (id != self.cntl_id and id != self.cntr_id)]
        probs_na = [probs[idx][id] for idx, id in enumerate(toks)]

        res = {
            'seq_na': seq_na,
            'probs_na': probs_na
        }

        return res

 

@hydra.main(version_base=None, config_path="../config", config_name="inference")
def run(conf: DictConfig) -> None:

    # Read model checkpoint.
    print('Starting inference')
    start_time = time.time()
    inference = Inference(conf)
    inference.run_inference()
    elapsed_time = time.time() - start_time
    print(f'Finished in {elapsed_time:.2f}s')

if __name__ == '__main__':
    run()
