import torch
import pytorch_lightning as pl
import transformers
import numpy as np
import matplotlib.pyplot as plt
import pdb
from bertviz.bertviz import head_view
from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer, BertConfig, BertForSequenceClassification
import re, os, tqdm, requests
from datasets import load_dataset
import torch.nn as nn
import logging
import torchmetrics
import wandb 
from collections import OrderedDict
from torch import optim
from torch.utils.data import DataLoader
import argparse
import collections
from typing import *
import json
import pathlib

#https://github.com/HelloJocelynLu/t5chem/blob/main/t5chem/archived/MultiTask.py for more info
# collections.Counter(dataset["test"]["label"])

class SequenceDataset(torch.utils.data.Dataset):
    SEGMENT_NAMES = ["PROA", "PROB", "PROC", "PROD", "PROE"] #make global for this class
    THREE_LETTER_AAS = ["ALA","ARG","ASN","ASP","CYS","GLU","GLN","GLY","HSD","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"]
    ONE_LETTER_AAS = ["A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
    LIPID_HEAD_GROUPS = ["PC","PE","PG","PI","PS","PA","CL","SM","CHOL","OTHERS"]
    
    """Protein sequence dataset"""
    def __init__(self, inputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.utils.data.Dataset:
        self.inputs = inputs
        self.targets = targets
        super().__init__()

    def __len__(self):
        return len(self.targets) #train length...

    def __getitem__(self, idx):
        input_ids = self.inputs["input_ids"][idx] #B, L
        token_type_ids = self.inputs["token_type_ids"][idx] #B, L
        attention_mask = self.inputs["attention_mask"][idx] #B, L
        input_reformats = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}
        target_reformats = {"labels": self.targets[idx]}
        return input_reformats, target_reformats
    
    @staticmethod
    def input_tokenizer(proper_inputs: List[str], hparams: argparse.ArgumentParser):
        tokenizer=BertTokenizer.from_pretrained("Rostlab/prot_bert",do_lower_case=False, return_tensors="pt",cache_dir=hparams.load_model_directory)
        inputs = tokenizer.batch_encode_plus(proper_inputs,
                                      add_special_tokens=True,
                                      padding=True, truncation=True, return_tensors="pt",
                                      max_length=hparams.max_length, return_attention_masks=True) #SUPPORTS two PAIRs for now... !Tokenize inputs as a dict type of Tensors
#         print(tokenizer.vocab)
        #('[PAD]', 0), ('[UNK]', 1), ('[CLS]', 2), ('[SEP]', 3), ('[MASK]', 4), ('L', 5), ('A', 6), 
        #('G', 7), ('V', 8), ('E', 9), ('S', 10), ('I', 11), ('K', 12), ('R', 13), ('D', 14), ('T', 15), 
        #('P', 16), ('N', 17), ('Q', 18), ('F', 19), ('Y', 20), ('M', 21), ('H', 22), ('C', 23), ('W', 24), 
        #('X', 25), ('U', 26), ('B', 27), ('Z', 28), ('O', 29)]
        return inputs 
    
    @classmethod
    def from_json(cls, filename: str, hparams: argparse.ArgumentParser):
        """WIP: Move functions from BertNERTokenizer.py
        This is per file of trajectory...
        Must fix!"""
#         print(hparams)
        max_residue = hparams.max_residue
        
        assert os.path.split(filename)[-1].split(".")[-1] == "json", "not a json file!" #get extension
        with open(filename, "r") as f:
            data = json.load(f)
        assert isinstance(data, dict), "wrong data format!"
#         print(data)
        
        seq = list(data.keys()) #e.g. TYR-483-PROA; len(seq) = num_res
        split_txt = np.array(list(map(lambda inp: inp.split("-"), seq))) #List[tuple of RESNAME_RESID_SEGID] -> np.array
        duplicates = 10 #Fake duplicates for batches (i.e. num files)

        ##1. AA Letter Mapping
        AA = SequenceDataset.__dict__["THREE_LETTER_AAS"]
        aa = SequenceDataset.__dict__["ONE_LETTER_AAS"]
        three2one = {THREE:ONE for THREE, ONE in list(zip(AA,aa))}
        seq_parser = lambda seqs: list(map(lambda seq: ' '.join(list(map(lambda aa: three2one.get(aa, None), seq.split(" ") ))), seqs ))    #THREE LETTER -> ONE LETTER

        ##2. Lipid Index Mapping and Extracting Coefficients
        lips = SequenceDataset.__dict__["LIPID_HEAD_GROUPS"] #This specific order matters!
        lip2idx = {k:v for v, k in enumerate(lips)}
        idx2lip = {v:k for k, v in lip2idx.items()}
        def lip_index(data: dict):
            seq = list(data.keys()) #e.g. TYR-483-PROA
            seq_nonzero = [[[v if isinstance(v, list) else [v]*3 for (l, v) in data[s].items()] for s in seq]] #duplicates x num_res x 8 x 3
            return seq_nonzero
#         print(len(lip_data[0])) #For 1 data, [num_AA lists; each AA list has 8 lipid type tuples];;; #172
#         print(SequenceDataset.__dict__)
        segs = SequenceDataset.__dict__["SEGMENT_NAMES"] #use [SEP] for different segment!

        ##3. DATA PREPROCESSING for Multi-segment Files
        split_txt = np.tile(split_txt, (2,1)) #Multiseg-test
        split_txt[len(seq):len(seq)*2,2] = "PROB" #Multiseg-test
        split_txt[2*len(seq):,2] = "PROC" #Multiseg-test
        
        all_resnames, all_segnames, modified_slice = split_txt[:,0].tolist(), split_txt[:,2], []
        all_resnames = [' '.join(all_resnames)] #List[str] -> [str_with_space]
        all_resnames = seq_parser(all_resnames) #[str_with_space_3letter] -> [str_with_space_1letter]
        all_resnames = all_resnames[0].split(" ") #List[str]
        all_resnames = all_resnames + (max_residue - len(seq)) * ["[PAD]"] #WIP
        
        print(max_residue, len(all_resnames))
        
        assert np.isin(all_segnames, segs).all(), "all segnames must match..."
        start_idx = 0
        for idx, seg in enumerate(np.unique(all_segnames)):
            end_idx_p1 = np.sum(all_segnames == seg) + start_idx if idx != (np.unique(all_segnames).shape[0]-1) else None
            current_slice = all_resnames[slice(start_idx, end_idx_p1)] #+ ["[SEP]"] #need sentence pair encoding (SEP not needed!!)
            modified_slice.append(current_slice) #not extend but append!!!!
            start_idx = end_idx_p1
            
        proper_inputs = [[' '.join(mod) for mod in modified_slice]] if len(modified_slice) > 1 else [' '.join(mod) for mod in modified_slice] #[List[seq_wo_sep]] for batch_encode_plus
            
        ##4. Fake duplicates: WIP, must be fixed!
        proper_inputs = proper_inputs * duplicates #List[10 lists of sent pairs]
        lip_data = lip_index(data)  #[list(num_res, 8, 3)]
        lip_data = lip_data * duplicates
        lip_data = np.array(lip_data) #duplicates, num_res, 8, 3    
        pad_to_lip = np.zeros((max_residue - len(all_resnames), *lip_data.shape[-2:])) #(pad_to_lip, 8, 3)
        print(pad_to_lip.shape, (max_residue - len(seq)))
        pad_to_lip = np.broadcast_to(pad_to_lip, (len(proper_inputs), *pad_to_lip.shape)) #(duplicates, pad_to_lip, 8, 3)
        print(pad_to_lip.shape)

        lip_data = np.concatenate((lip_data, pad_to_lip), axis=1) #make (duplicates, num_res+pad_to_lip, 8, 3)

#         print(proper_inputs)
        inputs = SequenceDataset.input_tokenizer(proper_inputs, hparams)
        targets = lip_data
        print(inputs, targets.shape)
        
        return cls(inputs, targets)
    
    @classmethod
    def from_directory(cls, directory: Union[pathlib.Path, str], hparams: argparse.ArgumentParser):
        potential_files = os.listdir(directory)
        filtered_files = list(filter(lambda inp: os.path.splitext(inp)[1] == ".json", potential_files))
        resnum_list = SequenceDataset.residue_length_check(filtered_files)
#         max_residue = max(resnum_list)
        max_residue = 400
        hparams.max_residue: int = max_residue #set a new attribute for Argparser; maximum residue num across json files!
#         print(hparams.max_residue)
#         hparams.resnum_list_idx: List[int] = np.arange(len(resnum_list)) #set a new attribute for Argparser
#         print(potential_files, filtered_files)
        dataset_list = [SequenceDataset.from_json(one_file, hparams) for _, one_file in enumerate(filtered_files)]
        concat_dataset = torch.utils.data.ConcatDataset(dataset_list) #WIP; must deal with different resnum datasets!
        return concat_dataset
    
    @staticmethod
    def residue_length_check(filtered_files: List[str]):
        resnum_list = []
        for one_file in filtered_files:
            with open(one_file, "r") as f:
                data = json.load(f)
            resnum_list.append(len( list(data.keys()) ))
        return resnum_list
                
# class NERSequenceDataset(torch.utils.data.Dataset):
#     """Protein sequence dataset
#     WIP!"""
#     def __init__(self, inputs: Dict) -> torch.utils.data.Dataset:
#         super().__init__()
#         self.inputs, self.targets = self.process_inputs(inputs) #proper inputs and made
        
#     @staticmethod
#     def process_inputs(inputs: Dict):
#         nframes = len(inputs)
        
#         return inputs_formatted, targets
        
#     def __len__(self):
#         return len(self.inputs) #train length...

#     def __getitem__(self, idx):
#         input_ids = self.inputs["input_ids"][idx] #B, L
#         token_type_ids = self.inputs["token_type_ids"][idx] #B, L
#         attention_mask = self.inputs["attention_mask"][idx] #B, L
#         input_reformats = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}
#         target_reformats = {"labels": self.targets[idx]}
#         return input_reformats, target_reformats

#     @classmethod
#     def from_json(cls, filename: str):
#         """WIP: Move functions from BertNERTokenizer.py"""
#         assert os.path.split(filename)[-1].split(".")[-1] == "json", "not a json file!" #get extension
#         with open(filename, "r") as f:
#             data = json.load(filename)
#         assert isinstance(data, dict), "wrong data format!"
#         return cls(data)    
    
if __name__ == "__main__":
#     from train import get_args
#     hparam = get_args()
#     dataset = load_dataset("yarongef/human_proteome_triplets", cache_dir=hparam.load_data_directory)

#     # collections.Counter(dataset["test"]["label"])

#     class SequenceDataset(torch.utils.data.Dataset):
#         """Protein sequence dataset"""
#         def __init__(self, inputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.utils.data.Dataset:
#             super().__init__()
#             self.inputs = inputs
#             self.targets = targets
    
#         def __len__(self):
#             return len(self.targets) #train length...

#         def __getitem__(self, idx):
#             input_ids = self.inputs["input_ids"][idx] #B, L
#             token_type_ids = self.inputs["token_type_ids"][idx] #B, L
#             attention_mask = self.inputs["attention_mask"][idx] #B, L
#             input_reformats = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}
#             target_reformats = {"labels": self.targets[idx]}
#             return input_reformats, target_reformats
    pass
