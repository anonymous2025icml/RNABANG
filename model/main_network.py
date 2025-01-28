from torch import nn
import torch.nn.functional as F
from model import main_block, embedder


class MainNetwork(nn.Module):

    def __init__(self, model_conf, tokenizer):
        super(MainNetwork, self).__init__()
        self._model_conf = model_conf

        self.embedding_layer = embedder.Embedder(model_conf, tokenizer)
        self.main_model = main_block.MainBlock(model_conf)

    def forward(self, input_feats):
        
        s_na, s_aa = self.embedding_layer(input_feats)
        
        model_out = self.main_model(s_na, s_aa, input_feats['T_aa'], input_feats['tidx_na'], input_feats['ridx_aa'], input_feats['pad_na'], input_feats['pad_aa'], input_feats['ct_na'])

        log_probs = F.log_softmax(model_out['logits'], dim=-1)

        pred_out = {
            'logits': model_out['logits'],
            'log_probs': log_probs
        }

        return pred_out
    
    def forward_aa(self, input_feats):

        s_aa = self.embedding_layer.forward_aa(input_feats)

        s_aa = self.main_model.forward_aa(s_aa, input_feats['T_aa'], input_feats['ridx_aa'], input_feats['pad_aa'])

        return s_aa
    
    def forward_na(self, s_aa, input_feats):

        s_na = self.embedding_layer.forward_na(input_feats)

        model_out = self.main_model.forward_na(s_na, s_aa, input_feats['tidx_na'], input_feats['pad_na'], input_feats['pad_aa'], input_feats['ct_na'])

        log_probs = F.log_softmax(model_out['logits'], dim=-1)

        pred_out = {
            'logits': model_out['logits'],
            'log_probs': log_probs
        }

        return pred_out


