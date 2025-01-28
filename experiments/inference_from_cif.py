import os
import time
import tree
import argparse
from types import SimpleNamespace
import numpy as np
import torch
import logging
import pandas as pd
from datetime import datetime

from model import main_network
from data import utils as du
from data import cif_processor
from experiments import utils as eu
from data import data_loader
from data import tokenizer

from omegaconf import DictConfig, OmegaConf






class Inference:

    def __init__(
            self,
            conf: DictConfig,
        ):

        self._log = logging.getLogger(__name__)


        # Prepare configs.
        self._conf = conf
        self._inf_conf = conf.inference
        self._tok_conf = conf.tokenizer

        self.sample_mode = conf.inference.sample_mode
        self.max_len = conf.inference.max_len
        self.kp = conf.inference.kp
        self.nof_samples = conf.inference.nof_samples

        self.cif_input_path = conf.inference.input_cif

        # Set-up directories
        self._weights_path = self._inf_conf.model_path

        output_dir = self._inf_conf.output_dir
        if self._inf_conf.name is None:
            name_string = self.cif_input_path.split('/')[-1].split('.')[0]
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
        self._load_ckpt()
        
       
        

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
        self.processor = cif_processor.Processor(self.tokenizer)
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


        self._log.info(f'Starting inference')

        _ = self.inference_fn()


    

    def inference_fn(self):


        test_feats = self.processor.process_cif(self.cif_input_path)

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
                f.write(f'>{idx},{sampled_prob}\n')
                f.write(f'{seq_na_str_uncut}\n')

            with open(os.path.join(self._output_dir, 'seq_na.fasta'), 'a') as f:
                f.write(f'>{idx},{sampled_prob}\n')
                f.write(f'{seq_na_str}\n')
            
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



def run(args: argparse.Namespace) -> None:
    
    start_time = time.time()
    inference = Inference(args)
    inference.run_inference()
    elapsed_time = time.time() - start_time
    print(f'Finished in {elapsed_time:.2f}s')



if __name__ == '__main__':
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Run inference for a single cif file.")

    parser.add_argument('--name', type=str, default=None, help='Name')
    parser.add_argument('--seed', type=int, default=123, help='Seed')
    parser.add_argument('--sample_mode', type=str, default='topk', help='Sampling mode')
    parser.add_argument('--kp', type=int, default=4, help='kp parameter')
    parser.add_argument('--nof_samples', type=int, default=1, help='Number of sequences to sample')
    parser.add_argument('--max_len', type=int, default=50, help='Maximum allowed sequence length')
    parser.add_argument('--output_dir', type=str, default='./inference_outputs', help='Path to save the output results')
    parser.add_argument('--model_path', type=str, default='./ckpt/icml.pth', help='Path to the model checkpoint')
    parser.add_argument('--input_cif', type=str, default='', help='Path to the input data')



    parser.add_argument('--vocab_path', type=str, default='./data/tokenizer.json', help='Path to tokenizer')
    parser.add_argument('--vocab_size', type=int, default=None, help='Vocabulary size')


    # Parse the arguments from the command line
    args = parser.parse_args()

    conf = SimpleNamespace(
        inference=SimpleNamespace(name = args.name,
                                  seed = args.seed,
                                  sample_mode = args.sample_mode,
                                  kp = args.kp,
                                  nof_samples = args.nof_samples,
                                  max_len = args.max_len,
                                  output_dir=args.output_dir,
                                  model_path=args.model_path,
                                  input_cif=args.input_cif),
        tokenizer=SimpleNamespace(vocab_path=args.vocab_path,
                                  vocab_size=args.vocab_size)
    )

    # Run the main function with parsed arguments
    run(conf)
