import numpy as np
import json



class Tokenizer:
    def __init__(self, conf):
            
            self._restypes_aa = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', '<pad>']

            with open(conf.vocab_path, 'r') as file:
                data = json.load(file)

            vocab = list(data['model']['vocab'].keys())[:conf.vocab_size-1]

            self._restypes_na = sorted(vocab, key=len, reverse=True) + ['<eos>', '<cntl>', '<cntr>', '<pad>']

            self._restypes_order_na = {restype: i for i, restype in enumerate(self._restypes_na)}
            self._restypes_num_na = len(self._restypes_na)

            self._restypes_order_aa = {restype: i for i, restype in enumerate(self._restypes_aa)}
            self._restypes_num_aa = len(self._restypes_aa)



    @property
    def restypes_aa(self):
        return self._restypes_aa
    
    @property
    def restypes_na(self):
        return self._restypes_na
    
    @property
    def restypes_order_aa(self):
        return self._restypes_order_aa
    
    @property
    def restypes_order_na(self):
        return self._restypes_order_na
    
    @property
    def restypes_num_aa(self):
        return self._restypes_num_aa
    
    @property
    def restypes_num_na(self):
        return self._restypes_num_na
    
    @property
    def restypes_num_na(self):
        return self._restypes_num_na
    

    def split_na(self, seq):

        if seq == '':
            return []
        for token in self._restypes_na:
            start = seq.find(token)
            if start != -1:
                return self.split_na(seq[:start]) + [token] + self.split_na(seq[len(token)+start:])
    

    def encode_na(self, tokens):

        token_idxs = []

        for token in tokens:

            token_idx = self.restypes_order_na.get(token, self.restypes_num_na)

            token_idxs.append(token_idx)

        token_idxs = np.array(token_idxs)

        return token_idxs
    

    def tokenize_na(self, residues):

        seq = ''.join(residues)

        tokens = self.split_na(seq)

        encodes = self.encode_na(tokens)

        return encodes


    def tokenize_aa(self, residues):
        
        encodes = []

        for res in residues:
            
            restype_idx = self.restypes_order_aa.get(res, self.restypes_num_aa)

            encodes.append(restype_idx)

        encodes = np.array(encodes)

        return encodes