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
from curtsies import fmtfuncs as cf
import pickle

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
        return len(self.targets["labels"]) #train length...

    def __getitem__(self, idx):
        input_ids = self.inputs["input_ids"][idx] #B, L
        token_type_ids = self.inputs["token_type_ids"][idx] #B, L
        attention_mask = self.inputs["attention_mask"][idx] #B, L
        input_reformats = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}
        target_reformats = {"labels": torch.from_numpy(self.targets["labels"][idx]).type(torch.long),
                            "coeffs": torch.from_numpy(self.targets["coeffs"][idx]).type(torch.float32),
                            "target_invalid_lipids": torch.from_numpy(self.targets["target_invalid_lipids"].reshape(-1,)).type(torch.bool)}
        
        return input_reformats, target_reformats
    
    @staticmethod
    def input_tokenizer(proper_inputs: List[str], hparams: argparse.ArgumentParser):
        tokenizer=BertTokenizer.from_pretrained("Rostlab/prot_bert",do_lower_case=False, return_tensors="pt",cache_dir=hparams.load_model_directory)
        inputs = tokenizer.batch_encode_plus(proper_inputs,
                                      add_special_tokens=True,
                                      padding=True, truncation=True, return_tensors="pt",
                                      max_length=hparams.max_length) #SUPPORTS two PAIRs for now... !Tokenize inputs as a dict type of Tensors
#         print(tokenizer.vocab)
        #('[PAD]', 0), ('[UNK]', 1), ('[CLS]', 2), ('[SEP]', 3), ('[MASK]', 4), ('L', 5), ('A', 6), 
        #('G', 7), ('V', 8), ('E', 9), ('S', 10), ('I', 11), ('K', 12), ('R', 13), ('D', 14), ('T', 15), 
        #('P', 16), ('N', 17), ('Q', 18), ('F', 19), ('Y', 20), ('M', 21), ('H', 22), ('C', 23), ('W', 24), 
        #('X', 25), ('U', 26), ('B', 27), ('Z', 28), ('O', 29)]
        return inputs 
    
    @staticmethod
    def from_json(filename: str, hparams: argparse.ArgumentParser):
        """DONE: WIP: Move functions from BertNERTokenizer.py
        This is per file of trajectory to be called inside
        DEF from_directory...!!
        Call 0. datapreprocessing
             0.1 AA_letter_mapper
             1. lipid_mapper
             2. pad_AA_lipid_dataset
        """
#         global max_residue
        max_residue = hparams.max_residue
        augment = hparams.augment #integer
        
        assert os.path.split(filename)[-1].split(".")[-1] == "json", "not a json file!" #get extension
        with open(filename, "r") as f:
            data = json.load(f)
        assert isinstance(data, dict), "wrong data format!"        
        masked_lipids = SequenceDataset.check_unique_lipid(data) #Get valid lipid types from trajectory; shape (duplicates, (num_res OR augment), 8)

        if augment:
            seq_original = list(data.keys()) #e.g. TYR-483-PROA; len(seq) = num_res
            #Below: starting key selection index
            dataset_list = []
            for start_idx in range(len(seq_original) - augment + 1):
                end_idx = start_idx + augment
                seq_augment = seq_original[start_idx:end_idx]
                data_augment = {k:data[k] for k in seq_augment} #select key-value from original data
                dataset_augment = SequenceDataset.parse_json_file(data_augment, masked_lipids=masked_lipids) #(1,augment_length) for one dataset
                dataset_list.append(dataset_augment)
            dataset = torch.utils.data.ConcatDataset(dataset_list) #(N,augment_length) for concatenated augmented dataset
        else:
            dataset = SequenceDataset.parse_json_file(data, masked_lipids=masked_lipids) #(1,padded_seq) for one dataset
            
        return dataset
                
    @classmethod
    def parse_json_file(cls, data: dict, return_lip_data=False, masked_lipids: np.ndarray=None):
        """
         Called inside from_json for normal/augmentation
         Call 0. datapreprocessing
         0.1 AA_letter_mapper
         1. lipid_mapper
         2. pad_AA_lipid_dataset
        """
        seq = list(data.keys()) #e.g. TYR-483-PROA; len(seq) = num_res OR augment_length 
        split_txt = np.array(list(map(lambda inp: inp.split("-"), seq))) #List[tuple of RESNAME_RESID_SEGID] -> np.array
        duplicates = 1 #Fake duplicates for batches (i.e. num files)
        if not return_lip_data: print(cf.red(f"Max num residues: {max_residue} Original length: {len(seq)}..."))

        all_resnames, proper_inputs = SequenceDataset.data_preprocessing(split_txt, seq)
        proper_inputs = proper_inputs * duplicates #List[duplicate lists of sent standalone AND/OR pairs]

        lip_data = SequenceDataset.lipid_mapper(data)  #[list(num_res, 8, 3)]
        lip_data = lip_data * duplicates
        lip_data = np.array(lip_data) #duplicates, (num_res OR augment), 8, 3  
        
        if return_lip_data:
            return lip_data #a Dataset instance OR lip_data mask!
        else:
            inputs, targets = SequenceDataset.pad_AA_lipid_dataset(all_resnames, lip_data, proper_inputs, masked_lipids)
            one_file_dataset = cls(inputs, targets)
            return one_file_dataset  #a Dataset instance OR lip_data mask!
    
    @staticmethod
    def check_unique_lipid(data: dict):
        lip_data: np.ndarray = SequenceDataset.parse_json_file(data, return_lip_data=True) #(1,padded_seq) for one dataset
        assert lip_data.ndim == 4, "dimension of lipid array data is wrong..."
        masked_lip_data = ~(np.apply_along_axis(func1d=lambda inp: np.sum(inp**2), axis=-1, arr=lip_data) == 0.) # --> duplicates, (num_res OR augment), 8
#         print(masked_lipids)
        return np.any(masked_lip_data, axis=1) ## --> duplicates, 8 #non-existing lipids will be False; else True (shape: duplicates, (num_res OR augment), 8) and (duplicates, 8) each
    
    @staticmethod
    def data_preprocessing(split_txt: np.ndarray, seq: List[str]):
        ##0. DATA PREPROCESSING for Multi-segment Files
        ##WARNING: max_residue is a global keyword 
        split_txt = np.tile(split_txt, (1,1)) #Multiseg-test
#         split_txt[len(seq):len(seq)*2,2] = "PROB" #Multiseg-test
#         split_txt[2*len(seq):,2] = "PROC" #Multiseg-test
        
        all_resnames, all_segnames, modified_slice = split_txt[:,0].tolist(), split_txt[:,2], []
        all_resnames = [' '.join(all_resnames)] #List[str] -> [str_with_space];; WIP: is this necessary for AA_letter_mapper???
        all_resnames = SequenceDataset.AA_letter_mapper(all_resnames) #[str_with_space_3letter] -> [str_with_space_1letter]
        all_resnames = all_resnames[0].split(" ") #List[str]
        all_resnames = all_resnames + (max_residue - len(all_resnames)) * ["[PAD]"] #WIP; type An AA List (e.g. ["A","G"...]
        assert len(np.where(np.array(all_resnames)!="[PAD]")[0]) == len(seq), cf.red("There is something wrong with sequence parsing...") #for np.where, must use np.ndarray !
        segs = SequenceDataset.__dict__["SEGMENT_NAMES"] #use [SEP] for different segment!
        assert np.isin(all_segnames, segs).all(), "all segnames must match..."
        
        start_idx = 0
        for idx, seg in enumerate(np.unique(all_segnames)):
            end_idx_p1 = np.sum(all_segnames == seg) + start_idx if idx != (np.unique(all_segnames).shape[0]-1) else None
            current_slice = all_resnames[slice(start_idx, end_idx_p1)] #+ ["[SEP]"] #need sentence pair encoding (SEP not needed!!)
            modified_slice.append(current_slice) #not extend but append!!!!
            start_idx = end_idx_p1
            
        proper_inputs = [[' '.join(mod) for mod in modified_slice]] if len(modified_slice) > 1 else [' '.join(mod) for mod in modified_slice] #[List[seq_wo_sep]] for batch_encode_plus
        return all_resnames, proper_inputs #take multi-segments into account
    
    @staticmethod
    def AA_letter_mapper(all_resnames: List[str]):     
        ##1. AA Letter Mapping
        AA = SequenceDataset.__dict__["THREE_LETTER_AAS"]
        aa = SequenceDataset.__dict__["ONE_LETTER_AAS"]
        three2one = {THREE:ONE for THREE, ONE in list(zip(AA,aa))}
        seq_parser = lambda seqs: list(map(lambda seq: ' '.join(list(map(lambda aa: three2one.get(aa, None), seq.split(" ") ))), seqs ))    #THREE LETTER -> ONE LETTER
        return seq_parser(all_resnames)
    
    @staticmethod
    def lipid_mapper(data: "json file read"):
        ##2. Lipid Index Mapping and Extracting Coefficients
        lips = SequenceDataset.__dict__["LIPID_HEAD_GROUPS"] #This specific order matters!
        lip2idx = {k:v for v, k in enumerate(lips)} #not needed since only coeffs are passed
        idx2lip = {v:k for k, v in lip2idx.items()} #not needed since only coeffs are passed
        def lip_index(data: dict):
            seq = list(data.keys()) #e.g. TYR-483-PROA
            seq_nonzero = [[[v if isinstance(v, list) else [v]*3 for (l, v) in data[s].items()] for s in seq]] #to make duplicates x (num_res OR augment) x 8 x 3
            return seq_nonzero
#         print(len(lip_data[0])) #For 1 data, [num_AA lists; each AA list has 8 lipid type tuples];;; #172
#         print(SequenceDataset.__dict__)
        return lip_index(data)

    @staticmethod
    def pad_AA_lipid_dataset(all_resnames: List[str], lip_data: np.ndarray, proper_inputs: List[str], masked_lipids: np.ndarray):
        ##3. all_resnames is already padded! to make dataset!
        pad_to_lip = np.zeros((len(all_resnames) - lip_data.shape[1], *lip_data.shape[-2:])) #(pad_to_lip, 8, 3)
        pad_to_lip = np.broadcast_to(pad_to_lip, (len(proper_inputs), *pad_to_lip.shape)) #(duplicates, pad_to_lip, 8, 3)
        lip_data = np.concatenate((lip_data, pad_to_lip), axis=1) #make (duplicates, (num_res OR augment)+pad_to_lip, 8, 3) = (duplicates, padde_sequence, 8, 3)
        inputs = SequenceDataset.input_tokenizer(proper_inputs, hparams)
        lip_labels = ~(np.apply_along_axis(func1d=lambda inp: np.sum(inp**2), axis=-1, arr=lip_data) == 0.) # --> duplicates, (num_res OR augment), 8
        targets = {"labels": lip_labels, "coeffs": lip_data, "target_invalid_lipids": masked_lipids} #lip_data (coeff); lip_labels (labels)
        return inputs, targets
    
    @staticmethod
    def from_directory(directory: Union[pathlib.Path, str], hparams: argparse.ArgumentParser):
        potential_files = os.listdir(directory)
        filtered_files = list(filter(lambda inp: os.path.splitext(inp)[1] == ".json", potential_files))
        resnum_list = SequenceDataset.residue_length_check(filtered_files)
        global max_residue #works well!
        assert hparams.augment <= min(resnum_list), "augment slicing must be smaller than minimum residue length..."
        max_residue = max(resnum_list) if not hparams.augment else hparams.augment #if augmenting, use augmentation length instead of sequence length
        print(cf.on_yellow(f"Maximum length to pad sequence is {max_residue}..."))
#         max_residue = 400
        hparams.max_residue: int = max_residue #set a new attribute for Argparser; maximum residue num across json files!
        dataset_list = [SequenceDataset.from_json(one_file, hparams) for _, one_file in enumerate(filtered_files)] #to concat dataset
        concat_dataset = torch.utils.data.ConcatDataset(dataset_list) #DONE with PADDING: WIP; must deal with different resnum datasets!
        
        if hparams.augment:
            print(cf.on_blue(f"By augmentation, dataset size has been enriched by {100*(len(concat_dataset)-len(filtered_files))/len(filtered_files)} percent..."))
        
        if hparams.save_to_file:
            save_to_file = os.path.splitext(hparams.save_to_file)[0] + ".pickle"
#             torch.save(concat_dataset, f"{save_to_file}")
            with open(save_to_file, "wb") as f:
                pickle.dump(concat_dataset, f)
        return concat_dataset
    
    @staticmethod
    def load_saved_dataset(filename: str):
        #How to solve pickle's can't find Attribute error?
        #https://stackoverflow.com/questions/27732354/unable-to-load-files-using-pickle-and-multiple-modules
#         class CustomUnpickler(pickle.Unpickler):
#             def find_class(self, module, name):
#                 if name == 'SequenceDataset':
#                     from dataset import SequenceDataset
#                     return SequenceDataset
#                 return super().find_class(module, name)
        from dataset import SequenceDataset
        filename = os.path.splitext(filename)[0] + ".pickle"
#         dataset = CustomUnpickler(open(f"{filename}","rb")).load()
        f = open(f"{filename}","rb")
        dataset = pickle.load(f)
        return dataset
    
    @staticmethod
    def residue_length_check(filtered_files: List[str]):
        #for getting max_residue list
        resnum_list = []
        for one_file in filtered_files:
            with open(one_file, "r") as f:
                data = json.load(f)
            resnum_list.append(len( list(data.keys()) ))
        return resnum_list
                
    
if __name__ == "__main__":
    from train import get_args
    hparams = get_args()

    ds2 = SequenceDataset.from_directory("/Scr/hyunpark/DL_Sequence_Collab/ProtLIpInt", hparams) #concat dataset instance
    dl = torch.utils.data.DataLoader(ds2, batch_size=15)
    one_ds = iter(dl).next()
    print(len(ds2))
    print(one_ds[0]["input_ids"], one_ds[1]["labels"].shape, one_ds[1]["target_invalid_lipids"]) #inputs, targets
