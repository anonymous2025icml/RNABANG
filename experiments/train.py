import os
import torch
import time
import numpy as np
import copy
import hydra
import wandb
import logging
import copy
import psutil

from collections import defaultdict
from collections import deque
from datetime import datetime
from omegaconf import DictConfig
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


from data import data_loader
from data import utils as du
from data import tokenizer
from experiments import utils as eu
from model import main_network



class Experiment:

    def __init__(
            self,
            *,
            conf: DictConfig,
        ):
        """Initialize experiment.

        Args:
            exp_cfg: Experiment configuration.
        """
        self._log = logging.getLogger(__name__)

        # Configs
        self._conf = conf
        self._exp_conf = conf.experiment
        if HydraConfig.initialized() and 'num' in HydraConfig.get().job:
            self._exp_conf.name = (f'{self._exp_conf.name}_{HydraConfig.get().job.num}')
        
        self._model_conf = conf.model
        self._data_conf = conf.data
        self._tok_conf = conf.tokenizer

        self.tokenizer = tokenizer.Tokenizer(self._tok_conf)

        self._use_wandb = self._exp_conf.use_wandb
        self._use_ddp = self._exp_conf.use_ddp

        if self._use_ddp :
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            dist.init_process_group(backend='nccl')
            self.ddp_info = eu.get_ddp_info()
            self._log.info(f"GPU {self.ddp_info['local_rank']} is connected")
            if self.ddp_info['rank'] not in [0,-1]:
                self._log.addHandler(logging.NullHandler())
                self._log.setLevel("ERROR")
                self._use_wandb = False
                self._exp_conf.ckpt_dir = None

        self.trained_steps = 0
        self.trained_epochs = 0
        
        # Warm starting
        ckpt_model_na = None
        if conf.experiment.warm_start_na:
            ckpt_path = conf.experiment.warm_start_na
            self._log.info(f'Warm starting na from: {ckpt_path}')
            ckpt_pkl = eu.read_pkl(ckpt_path, use_torch=True)
            ckpt_model_na = ckpt_pkl['model']

        ckpt_model_aa = None
        if conf.experiment.warm_start_aa:
            ckpt_path = conf.experiment.warm_start_aa
            self._log.info(f'Warm starting aa from: {ckpt_path}')
            ckpt_pkl = eu.read_pkl(ckpt_path, use_torch=True)
            ckpt_model_aa = ckpt_pkl['model']
            ckpt_model_aa['embedding_layer.residue_embedder_aa.0.weight'] = ckpt_model_aa['embedding_layer.residue_embedder_aa.0.weight'][:-1]
            _ = ckpt_model_aa.pop('main_model.log_head.weight')
            _ = ckpt_model_aa.pop('main_model.log_head.bias')

        # Initialize experiment objects
        self._model = main_network.MainNetwork(self._model_conf, self.tokenizer)

        if ckpt_model_na is not None:
            self._model.load_state_dict(ckpt_model_na, strict=False)

        if ckpt_model_aa is not None:
            self._model.load_state_dict(ckpt_model_aa, strict=False)

        # GPU mode
        

        if not self._use_ddp:
            if torch.cuda.is_available() and self._exp_conf.use_gpu:
                self.device = f"cuda:0"
            else:
                self.device = 'cpu'
            self._model = self.model.to(self.device)
            self._log.info(f"Using device: {self.device}")
        else:
            self.device = torch.device("cuda",self.ddp_info['local_rank'])
            model = self.model.to(self.device)
            self._model = DDP(model, device_ids=[self.ddp_info['local_rank']], output_device=self.ddp_info['local_rank'])
            self._log.info(f"Multi-GPU training on GPUs in DDP mode, node_id : {self.ddp_info['node_id']}")


        num_parameters = sum(p.numel() for p in self._model.parameters())
        num_trainable_parameters = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        self._exp_conf.num_parameters = num_parameters
        self._log.info(f'Number of model parameters {num_parameters} ({num_trainable_parameters} trainable)')

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._exp_conf.learning_rate, betas=(0.9, 0.999), eps=1e-06, amsgrad=True)
        self._scheduler = torch.optim.lr_scheduler.LambdaLR(self._optimizer, lr_lambda=lambda step: eu.get_lr_lambda_gamma(step, self._exp_conf.warmup_steps))

        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none", label_smoothing=0.0)

        dt_string = datetime.now().strftime("%dD_%mM_%YY_%Hh_%Mm_%Ss")
        if self._exp_conf.ckpt_dir is not None:
            # Set-up checkpoint location
            ckpt_dir = os.path.join(
                self._exp_conf.ckpt_dir,
                self._exp_conf.name,
                dt_string)
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir, exist_ok=True)
            self._exp_conf.ckpt_dir = ckpt_dir
            self._log.info(f'Checkpoints saved to: {ckpt_dir}')
        else:  
            self._log.info('Checkpoint not being saved.')
            
            
        self._aux_data_history = deque(maxlen=100)


    @property
    def model(self):
        return self._model

    @property
    def conf(self):
        return self._conf
    
    def init_wandb(self):
        self._log.info('Initializing Wandb.')
        conf_dict = OmegaConf.to_container(self._conf, resolve=True)
        wandb.init(
            project='rsgm',
            name=self._exp_conf.name,
            config=dict(eu.flatten_dict(conf_dict)),
            dir=self._exp_conf.wandb_dir,
        )
        self._exp_conf.run_id = wandb.util.generate_id()
        self._exp_conf.wandb_dir = wandb.run.dir
        self._log.info(
            f'Wandb: run_id={self._exp_conf.run_id}, run_dir={self._exp_conf.wandb_dir}')

    def create_dataset(self):

        train_csv, valid_csv = eu.create_split(self._conf, self._log)
        if self._exp_conf.ckpt_dir is not None:
            valid_csv.to_csv(os.path.join(self._exp_conf.ckpt_dir, 'valid_samples.csv'), index=False)

        # Datasets
        train_dataset = data_loader.Dataset(
            train_csv,
            self._data_conf,
            self.tokenizer,
            is_training=True,
        )

        valid_dataset = data_loader.Dataset(
            valid_csv,
            self._data_conf,
            self.tokenizer,
            is_training=True,
        )

        # Samplers
        if not self._use_ddp:
            train_sampler = data_loader.Sampler(
                data_conf=self._data_conf,
                dataset=train_dataset,
                sample_mode=self._exp_conf.sample_mode,
                sample_num=self._exp_conf.sample_num
            )
            valid_sampler = data_loader.Sampler(
                data_conf=self._data_conf,
                dataset=valid_dataset,
                sample_mode=self._exp_conf.sample_mode,
                sample_num=self._exp_conf.sample_num
            )
        else:
            train_sampler = data_loader.DistributedSampler(
                data_conf=self._data_conf,
                dataset=train_dataset,
                sample_mode=self._exp_conf.sample_mode,
                sample_num=self._exp_conf.sample_num,
                num_replicas = self.ddp_info['world_size'],
                rank = self.ddp_info['rank']
            )
            valid_sampler = data_loader.DistributedSampler(
                data_conf=self._data_conf,
                dataset=valid_dataset,
                sample_mode=self._exp_conf.sample_mode,
                sample_num=self._exp_conf.sample_num,
                num_replicas = self.ddp_info['world_size'],
                rank = self.ddp_info['rank']
            )

        # Loaders
        num_workers = self._exp_conf.num_loader_workers
        train_loader = du.create_data_loader(
            train_dataset,
            self.tokenizer,
            sampler=train_sampler,
            batch_size=self._exp_conf.batch_size if not self._use_ddp else self._exp_conf.batch_size // self.ddp_info['world_size'],
            num_workers=num_workers,
        )
        valid_loader = du.create_data_loader(
            valid_dataset,
            self.tokenizer,
            sampler=valid_sampler,
            batch_size=self._exp_conf.batch_size if not self._use_ddp else self._exp_conf.batch_size // self.ddp_info['world_size'],
            num_workers=num_workers,
        )

        return train_loader, valid_loader, train_sampler, valid_sampler



    def start_training(self):

        if self._use_wandb: self.init_wandb()
        
        self._model.train()
        self._optimizer.zero_grad()

        train_loader, valid_loader, train_sampler, valid_sampler = self.create_dataset()

        logs = []

        for epoch in range(self.trained_epochs, self._exp_conf.num_epoch):

            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            if valid_sampler is not None:
                valid_sampler.set_epoch(epoch)

            self.trained_epochs = epoch
    
            epoch_log = self.train_epoch(train_loader, valid_loader)

        self._log.info('Done')
        

    def train_epoch(self, train_loader, valid_loader):
        log_losses = defaultdict(list)
        global_logs = []
        log_time = time.time()
        step_time = time.time()

        # Training
        for train_feats, sample_ids in train_loader:

            train_feats = {key: value.to(self.device) for key, value in train_feats.items()}
            
            loss, aux_data = self.update_fn(train_feats)
            
            if torch.isnan(loss):
                raise Exception(f'NaN encountered')

            for k,v in aux_data.items():
                log_losses[k].append(v)

            self.trained_steps += 1

            # Logging to terminal train loss
            if self.trained_steps == 1 or self.trained_steps % self._exp_conf.log_freq == 0:

                elapsed_time = time.time() - log_time
                log_time = time.time()
                step_per_sec = self._exp_conf.log_freq / elapsed_time

                rolling_losses = {key: torch.stack(value).sum() for key, value in log_losses.items()}
                
                if self._use_ddp:
                    for key, value in rolling_losses.items():
                        dist.all_reduce(value, op=dist.ReduceOp.SUM)

                rolling_losses = eu.divide_dict(rolling_losses)

                loss_log = ' '.join([f'{k}={v:.4f}' for k,v in rolling_losses.items() if 'batch' not in k])

                self._log.info(f'[Train {self.trained_steps}]: {loss_log}, steps/sec={step_per_sec:.5f}')

                del log_losses, rolling_losses

                log_losses = defaultdict(list)

        # Validating
        self._log.info(f'End of training for epoch {self.trained_epochs+1}!')

        log_losses_valid = defaultdict(list)
        log_val_time = time.time()

        for valid_feats, sample_ids in valid_loader:

            valid_feats = {key: value.to(self.device) for key, value in valid_feats.items()}
            
            self._model.eval()

            for m in self._model.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()

            with torch.no_grad():
                loss, aux_data = self.loss_fn(valid_feats)

            for k,v in aux_data.items():
                log_losses_valid[k].append(v)

        # Logging to terminal validation loss
        elapsed_time = time.time() - log_val_time
        step_per_sec = len(valid_loader) / elapsed_time

        rolling_losses = {key: torch.stack(value).sum() for key, value in log_losses_valid.items()}

        if self._use_ddp:
            for key, value in rolling_losses.items():
                dist.all_reduce(value, op=dist.ReduceOp.SUM)

        rolling_losses = eu.divide_dict(rolling_losses)

        loss_log = ' '.join([f'{k}={v:.4f}' for k,v in rolling_losses.items() if 'batch' not in k])

        self._log.info(f'[Validation {self.trained_epochs+1}]: {loss_log}, steps/sec={step_per_sec:.5f}')

        del log_losses_valid, rolling_losses

        log_losses_valid = defaultdict(list)

        # Take checkpoint
        if self._exp_conf.ckpt_dir is not None:
            ckpt_path = os.path.join(self._exp_conf.ckpt_dir, f'step_{self.trained_steps}.pth')

            self._ckpt_path = ckpt_path

            eu.write_checkpoint(
                ckpt_path,
                copy.deepcopy(self.model.state_dict()),
                self._conf,
                copy.deepcopy(self._optimizer.state_dict()),
                self.trained_epochs,
                self.trained_steps,
                logger=self._log,
                use_torch=True
            )
        
    
    def update_fn(self, data):

        self._model.train()

        self._optimizer.zero_grad()

        loss, aux_data = self.loss_fn(data)

        loss.backward()
        
        self._optimizer.step()

        self._scheduler.step()

        return loss, aux_data


    def loss_fn(self, batch):

        model_out = self.model(batch)

        batch_size, num_res_na, num_c_na = model_out['logits'].shape

        pad_mask = batch['pad_na'][:,2:]

        mask = pad_mask

        logits = model_out['logits'][:,1:-1].reshape(-1, num_c_na)

        target = (batch['ttar_na'][:,:-2]*pad_mask).reshape(-1)
        
        loss = self.cross_entropy(logits, target).reshape(batch_size, num_res_na-2)

        final_loss = (loss*mask).sum(-1) / mask.sum(-1)
        
        aux_data = {
            'total_loss': final_loss.sum(),
            'total_samples': batch['rna'].sum()+(1-batch['rna']).sum(),
            'rna_loss': (batch['rna']*final_loss).sum(),
            'rna_samples': batch['rna'].sum(),
            'dna_loss': ((1-batch['rna'])*final_loss).sum(),
            'dna_samples': (1-batch['rna']).sum()
        }
        
        return final_loss.sum()/batch_size, aux_data
    




@hydra.main(version_base=None, config_path="../config", config_name="base")
def run(conf: DictConfig) -> None:

    exp = Experiment(conf=conf)
    exp.start_training()
    


if __name__ == '__main__':
    run()
