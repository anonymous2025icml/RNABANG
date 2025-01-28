import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


from model.encoder import EncoderAA, EncoderNA




class MainBlock(nn.Module):

    def __init__(self, model_conf):
        super(MainBlock, self).__init__()
        self.model_conf = model_conf

        self.trunk = nn.ModuleDict()

        for b in range(self.model_conf.num_blocks_aa):
            self.trunk[f'encoder_aa_{b}'] = EncoderAA(self.model_conf)

        for b in range(self.model_conf.num_blocks_na):
            self.trunk[f'encoder_na_{b}'] = EncoderNA(self.model_conf)

        self.log_head = nn.Linear(self.model_conf.c_s, self.model_conf.c_lm_head)

        
    def forward(self, s_na, s_aa, T_aa, s_na_pos, s_aa_pos, pad_na, pad_aa, ct_na):
        
        for b in range(self.model_conf.num_blocks_aa):
            
            s_aa = self.trunk[f'encoder_aa_{b}'](s_aa, s_aa_pos, T_aa, pad_aa)

        for b in range(self.model_conf.num_blocks_na):
            
            s_na = self.trunk[f'encoder_na_{b}'](s_na, s_aa, s_na_pos, pad_na, pad_aa, ct_na)
        
        logits_na = self.log_head(s_na)

        model_out = {
            'logits': logits_na
        }

        return model_out
    

    def forward_aa(self, s_aa, T_aa, s_aa_pos, pad_aa):

        for b in range(self.model_conf.num_blocks_aa):
            
            s_aa = self.trunk[f'encoder_aa_{b}'](s_aa, s_aa_pos, T_aa, pad_aa)

        return s_aa
        

    def forward_na(self, s_na, s_aa, s_na_pos, pad_na, pad_aa, ct_na):

        for b in range(self.model_conf.num_blocks_na):
            
            s_na = self.trunk[f'encoder_na_{b}'](s_na, s_aa, s_na_pos, pad_na, pad_aa, ct_na)
        
        logits_na = self.log_head(s_na)

        model_out = {
            'logits': logits_na
        }

        return model_out
