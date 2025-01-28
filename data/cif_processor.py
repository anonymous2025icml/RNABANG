import math
from typing import Optional

import torch
import torch.distributed as dist

import os
import numpy as np
import torch
import pandas as pd
import gemmi

from data import utils as du
from data import rigid_utils








class Processor():
    def __init__(
            self,
            tokenizer
        ):

        self.scale_pos = lambda x: x * 0.1
        self.scale_rigids = lambda x: x.apply_trans_fn(self.scale_pos)

        self.tokenizer = tokenizer



    def process_cif(self, cif_path):

        struct = gemmi.read_structure(cif_path)
        struct.remove_alternative_conformations()

        chain = struct[0]['A']
        
        for residue in chain:
            for atom in residue:
                if atom.name in ['CA', 'N', 'C', 'CB']:
                    if atom.occ < 1.0 and not atom.has_altloc() and residue.het_flag == 'A':
                        raise Exception("Ambigous protein structure")
                    
        ids = []
        residues = []
        coords = []

        for r in chain:
            n,ca,c = [None],[None],[None]
            try:
                n = np.array(r.sole_atom("N").pos.tolist())
                ca = np.array(r.sole_atom("CA").pos.tolist())
                c = np.array(r.sole_atom("C").pos.tolist())
            except:
                pass

            if n[0]==None or ca[0]==None or c[0]==None:
                    continue
            else:
                ids.append(r.seqid.num)
                residues.append(r.name)
                coords.append(np.concatenate(du.rigidFrom3points(n, ca, c)))

        coords = np.array(coords)
        com = np.sum(coords, axis=0)[:3]/len(coords)
        coords[:,:3] -= com

        frames_aa = du.linear_to_4x4(coords)
        T_aa = torch.tensor(frames_aa).float()
        T_aa = rigid_utils.Rigid.from_tensor_4x4(T_aa.unsqueeze(1))[:, 0]
        T_aa = self.scale_rigids(T_aa)
        
        ridx_aa = np.array(ids)
        ridx_aa = ridx_aa - np.min(ridx_aa)

        rtype_aa = residues
        rtype_aa = self.tokenizer.tokenize_aa(rtype_aa)

        ttype_na = ['<cntl>', '<cntr>']
        tidx_na = [0, 1]
        cont_token = -1

        ttype_na = self.tokenizer.encode_na(ttype_na)

        final_feats = {
            'ttype_na': torch.tensor(ttype_na).to(torch.int)[None],
            'rtype_aa': torch.tensor(rtype_aa).to(torch.int)[None],
            'tidx_na': torch.tensor(tidx_na).to(torch.int16)[None],
            'ridx_aa': torch.tensor(ridx_aa).to(torch.int16)[None],
            'T_aa': T_aa,
            'ct_na': torch.tensor(cont_token+1).to(torch.int16)[None]
        }

        final_feats['pad_aa'] = torch.ones_like(final_feats['ridx_aa'])
        final_feats['pad_na'] = torch.ones_like(final_feats['tidx_na'])
        final_feats['rna'] = torch.tensor(1).to(torch.int)[None]
        
        
        
        return final_feats