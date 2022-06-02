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

#https://github.com/HelloJocelynLu/t5chem/blob/main/t5chem/archived/MultiTask.py for more info
# collections.Counter(dataset["test"]["label"])

class SequenceDataset(torch.utils.data.Dataset):
    """Protein sequence dataset"""
    def __init__(self, inputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.utils.data.Dataset:
        super().__init__()
        self.inputs = inputs
        self.targets = targets
    
    def __len__(self):
        return len(self.targets) #train length...

    def __getitem__(self, idx):
        input_ids = self.inputs["input_ids"][idx] #B, L
        token_type_ids = self.inputs["token_type_ids"][idx] #B, L
        attention_mask = self.inputs["attention_mask"][idx] #B, L
        input_reformats = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}
        target_reformats = {"labels": self.targets[idx]}
        return input_reformats, target_reformats

if __name__ == "__main__":
    from main import get_args
    hparam = get_args()
    dataset = load_dataset("yarongef/human_proteome_triplets", cache_dir=hparam.load_data_directory)

    # collections.Counter(dataset["test"]["label"])

    class SequenceDataset(torch.utils.data.Dataset):
        """Protein sequence dataset"""
        def __init__(self, inputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.utils.data.Dataset:
            super().__init__()
            self.inputs = inputs
            self.targets = targets
    
        def __len__(self):
            return len(self.targets) #train length...

        def __getitem__(self, idx):
            input_ids = self.inputs["input_ids"][idx] #B, L
            token_type_ids = self.inputs["token_type_ids"][idx] #B, L
            attention_mask = self.inputs["attention_mask"][idx] #B, L
            input_reformats = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}
            target_reformats = {"labels": self.targets[idx]}
            return input_reformats, target_reformats
