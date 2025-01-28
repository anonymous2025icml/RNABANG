import torch
from torch import nn
from data.tokenizer import Tokenizer
from model.primitives import Linear, Embedding, LayerNorm, RMSNorm


class Embedder(nn.Module):

    def __init__(self, model_conf, tokenizer):

        super(Embedder, self).__init__()

        self.tokenizer = tokenizer

        self.restypes_num_aa = self.tokenizer.restypes_num_aa
        self.restypes_num_na = self.tokenizer.restypes_num_na
        self._model_conf = model_conf

        pad_na = self.tokenizer.tokenize_na(['<pad>'])[0]
        pad_aa = self.tokenizer.tokenize_aa(['<pad>'])[0]

        res_embed_size = self._model_conf.c_s

        # Embedders
        self.residue_embedder_aa = nn.Sequential(
            Embedding(self.restypes_num_aa, res_embed_size, padding_idx = pad_aa, init='bietti'),
            RMSNorm(res_embed_size)
        )

        self.residue_embedder_na = Embedding(self.restypes_num_na, res_embed_size-2, padding_idx = pad_na, init='bietti')
        
        self.type_embedder_na = Embedding(2, 2, init='bietti')

        self.layer_norm = RMSNorm(res_embed_size)
       


        
    def forward(self, batch):
        
        s_aa_embed = self.residue_embedder_aa(batch['rtype_aa'])

        num_ts_na = batch['tidx_na'].shape[1]

        rna_marker = self.type_embedder_na(batch['rna']).unsqueeze(-2).tile(1,num_ts_na,1)
        
        s_na_embed = self.layer_norm(torch.cat((self.residue_embedder_na(batch['ttype_na']), rna_marker), dim=-1))


        return s_na_embed, s_aa_embed
    

    def forward_aa(self, batch):
        
        s_aa_embed = self.residue_embedder_aa(batch['rtype_aa'])

        return s_aa_embed
    

    def forward_na(self, batch):

        num_ts_na = batch['tidx_na'].shape[1]

        rna_marker = self.type_embedder_na(batch['rna']).unsqueeze(-2).tile(1,num_ts_na,1)
        
        s_na_embed = self.layer_norm(torch.cat((self.residue_embedder_na(batch['ttype_na']), rna_marker), dim=-1))

        return s_na_embed