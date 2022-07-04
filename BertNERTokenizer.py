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
import dataset as dl
import argparse
from typing import *
from torchcrf import CRF
import collections
import logging
import os
import re
import codecs
import unicodedata
from typing import List, Optional
from transformers import PreTrainedTokenizer
from SmilesPE.tokenizer import SPE_Tokenizer
from SmilesPE.pretokenizer import atomwise_tokenizer, kmer_tokenizer
import inspect
import functools
import json
import itertools
from dataset import  NERSequenceDataset

"""https://colab.research.google.com/drive/1tsiTpC4i26QNdRzBHFfXIOFVToE54-9b?usp=sharing#scrollTo=L8eapLvAiGeY"""

class BertNERTokenizer(torch.nn.Module):
    def __init__(self, ner_config):
        super().__init__()
        self.ner_config
        
    def batch_encode_plus(self, inputs: torch.Tensor, targets: torch.LongTensor):
        #inputs: B,residue_length; targets: B,residue_length
        pass
    
class FeaturizerNotExistentError(Exception):
    def __init__(self, ):
        msg = "This featurizer (i.e. goods: smiles/deepsmiles/selfies etc...) selection does not exist..."
        return super().__init__(msg)

class TokenizerNotExistentError(Exception):
    def __init__(self, ):
        msg = "This tokenizer (i.e. goods: atom/kmer/spe etc...) selection does not exist..."
        return super().__init__(msg)

class SPE_Tokenizer_Wrapped(SPE_Tokenizer):
    """To be consistent with other tokenizers..."""
    def __call__(self, smi, dropout=0):
        return self.tokenize(smi, dropout)

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab
    
class BertNERTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab_file,
        spe_file,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        which_featurizer="smiles",
        which_tokenizer="spe",
        **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )
        self.which_featurizer = which_featurizer
        self.which_tokenizer = which_tokenizer
        
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'.".format(vocab_file)
            )
        if not os.path.isfile(spe_file):
            raise ValueError(
                "Can't find a SPE vocabulary file at path '{}'.".format(spe_file)
            )
        self.vocab = load_vocab(vocab_file) #Covers both Atomwise and SPE; not KMER!
        self.spe_vocab = codecs.open(spe_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        
        if which_featurizer == "smiles":
            assert which_featurizer in spe_file.lower(), "Vocabulary learning was not done in SMILES format..."
        elif which_featurizer == "deepsmiles":
            assert which_featurizer in spe_file.lower(), "Vocabulary learning was not done in DEEPSMILES format..."
        elif which_tokenizer == "selfies":
            assert which_featurizer in spe_file.lower(), "Vocabulary learning was not done in SELFIES format..."
        else:
            raise FeaturizerNotExistentError()        
        
        if which_tokenizer == "spe":
            self.tokenizer = SPE_Tokenizer_Wrapped(self.spe_vocab)
        elif which_tokenizer == "atom":
            self.tokenizer = atomwise_tokenizer
#         elif which_tokenizer == "kmer":
#             #WIP Do not use KMER yet...; hard to build vocabulary
#             self.tokenizer = kmer_tokenizer
        else:
            raise TokenizerNotExistentError()

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text):
        return self.tokenizer(text).split(' ') if self.which_tokenizer == "spe" else self.tokenizer(text) #spe gives a string with spaces; others give a list

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A BERT sequence has the following format:
        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` method.
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True if the token list is already formatted with special tokens for the model
        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        A BERT sequence pair mask has the following format:
        ::
            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |
        if token_ids_1 is None, only returns the first portion of the mask (0's).
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of `token type IDs <../glossary.html#token-type-ids>`_ according to the given
            sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, vocab_path):
        """
        Save the sentencepiece vocabulary (copy original file) and special tokens file to a directory.
        Args:
            vocab_path (:obj:`str`):
                The directory in which to save the vocabulary.
        Returns:
            :obj:`Tuple(str)`: Paths to the files saved.
        """
        index = 0
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, VOCAB_FILES_NAMES["vocab_file"])
        else:
            vocab_file = vocab_path
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(vocab_file)
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)
      
if __name__ == "__main__":      
    with open("sample_output_coeffs.json", "r") as f:
        data = json.load(f)
    
#     seq = list(data.keys()) #e.g. TYR-483-PROA
#     seq_nonzero = list(map(lambda s: list(filter(lambda l: l, data[s].values())), seq)) #List[List of COEFF]
    def lip_index(data: dict):
        seq = list(data.keys()) #e.g. TYR-483-PROA
#         seq_nonzero = list(map(lambda s: list(filter(lambda l: l, data[s].values())), seq)) #List[List of COEFF]
        seq_nonzero = [[(l, v) for (l, v) in data[s] if isinstance(v, list)] for s in seq]
        return seq_nonzero
    print(lip_index(data))
    split_txt = np.array(list(map(lambda inp: inp.split("-"), seq))) #List[tuple of RESNAME_RESID_SEGID] -> np.array
    
    ##AA Letter Mapping
    AA = ["ALA","ARG","ASN","ASP","CYS","GLU","GLN","GLY","HSD","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"]
    aa = ["A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
    three2one = {THREE:ONE for THREE, ONE in list(zip(AA,aa))}
    seq_parser = lambda seqs: list(map(lambda seq: ' '.join(list(map(lambda aa: three2one.get(aa, None), seq.split(" ") ))), seqs ))    #THREE LETTER -> ONE LETTER
#     print(np.isin(split_txt[:,0], AA).all())

    ##Lipid index dictionary
    lips = ["PC","PE","PG","PI","PS","PA","CL","SM","CHOL","OTHERS"] #This specific order matters!
    lip2idx = {k:v for v, k in enumerate(lips)}
    idx2lip = {v:k for k, v in lip2idx.items()}
    
    segs = ["PROA","PROB","PROC","PROD"] #use [SEP] for different segment!
#     aa_seg = list(itertools.product(aa, seg))

    #DATA PREPROCESSING
#     split_txt = np.tile(split_txt, (2,1)) #Multiseg-test
#     split_txt[len(seq):len(seq)*2,2] = "PROB" #Multiseg-test
#     split_txt[2*len(seq):,2] = "PROC" #Multiseg-test
    all_resnames, all_segnames, modified_slice = split_txt[:,0].tolist(), split_txt[:,2], []
    all_resnames = [' '.join(all_resnames)] #List[str] -> [str_with_space]
#     print(all_resnames)
    all_resnames = seq_parser(all_resnames) #[str_with_space_3letter] -> [str_with_space_1letter]
    all_resnames = all_resnames[0].split(" ") #List[str]
#     print(all_resnames)

    assert np.isin(all_segnames, segs).all(), "all segnames must match..."
#     proper_inputs = []
    start_idx = 0
    for seg in np.unique(all_segnames):
        end_idx_p1 = np.sum(all_segnames == seg) + start_idx
        current_slice = all_resnames[slice(start_idx, end_idx_p1)] #+ ["[SEP]"] #need sentence pair encoding (SEP not needed!!)
#         print(current_slice)
        modified_slice.append(current_slice) #not extend but append!!!!
        start_idx = end_idx_p1
#         proper_inputs.append()
#     modified_slice = ["[CLS]"] + modified_slice #NOT necessary
#     modified_slice.pop() #last SEP token should be gone! #<SEQ1 + SEP + SEQ2 + SEP + SEQ3 ...>
    proper_inputs = [[' '.join(mod) for mod in modified_slice]] if len(modified_slice) > 1 else [' '.join(mod) for mod in modified_slice] #[List[seq_wo_sep]] for batch_encode_plus
#     print(proper_inputs)

    def get_args():
        parser = argparse.ArgumentParser(description='Training')

        #Model related
        parser.add_argument('--load-model-directory', "-dirm", type=str, default="/Scr/hyunpark/DL_Sequence_Collab/pfcrt_projects/output", help='This is where model is/will be located...')  
        parser.add_argument('--load-model-checkpoint', "-ckpt", type=str, default=None, help='which checkpoint...')  
        parser.add_argument('--model-name', type=str, default='Rostlab/prot_bert', help='HUGGINGFACE Backbone model name card')

        #Molecule (Dataloader) related
        parser.add_argument('--load-data-directory', "-dird", default="/Scr/hyunpark/DL_Sequence_Collab/data", help='This is where data is located...')  
        parser.add_argument('--dataset', type=str, default="yarongef/human_proteome_triplets", help='pass dataset...')  

        #Optimizer related
        parser.add_argument('--max-epochs', default=60, type=int, help='number of epochs max')
        parser.add_argument('--min-epochs', default=1, type=int, help='number of epochs min')
        parser.add_argument('--batch-size', '-b', default=2048, type=int, help='batch size')
        parser.add_argument('--learning-rate', '-lr', default=1e-3, type=float, help='learning rate')
        parser.add_argument('--ngpus', type=int, default=-1, help='Number of GPUs, -1 use all available. Use CUDA_VISIBLE_DEVICES=1, to decide gpus')
        parser.add_argument('--num-nodes', type=int, default=1, help='Number of nodes')
        parser.add_argument('--warm-up-split', type=int, default=5, help='warmup times')
        parser.add_argument('--scheduler', type=str, default="cosine", help='scheduler type')
        parser.add_argument('--accelerator', "-accl", type=str, default="gpu", help='accelerator type')

        #Misc.
        parser.add_argument('--seed', type=int, default=42, help='seeding number')
        parser.add_argument('--precision', type=int, default=32, choices=[16, 32], help='Floating point precision')
        parser.add_argument('--monitor', type=str, default="val_loss_mean", help='metric to watch')
        parser.add_argument('--loss', '-l', type=str, default="classification", choices=['classification', 'contrastive', 'ner'], help='loss for training')
        parser.add_argument('--save_top_k', type=int, default="5", help='num of models to save')
        parser.add_argument('--patience', type=int, default=10, help='patience for stopping')
        parser.add_argument('--metric_mode', type=str, default="min", help='mode of monitor')
        parser.add_argument('--distributed-backend', default='ddp', help='Distributed backend: dp, ddp, ddp2')
        parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data prefetch')
        parser.add_argument('--amp-backend', type=str, default="native", help='Torch vs NVIDIA AMP')
        parser.add_argument('--max_length', type=int, default=1536, help='length for padding seqs')
        parser.add_argument('--label-smoothing', '-ls', type=float, default=0., help='CE loss regularization')
        parser.add_argument('--sanity-checks', '-sc', type=int, default=2, help='Num sanity checks..')
        parser.add_argument('--z_dim', '-zd', type=int, default=1024, help='CURL purpose.., SAME as self.encoder_features')
        parser.add_argument('--ner', '-ner', type=bool, default=False, help='NER training')
        parser.add_argument('--ner-config', '-nc', type=str, default=None, help='NER config')

        args = parser.parse_args()
        return args

    hparams = get_args()
    tokenizer=BertTokenizer.from_pretrained("Rostlab/prot_bert",do_lower_case=False, return_tensors="pt",cache_dir=hparams.load_model_directory)
    inputs = tokenizer.batch_encode_plus(proper_inputs,
                                  add_special_tokens=True,
                                  padding=True, truncation=True, return_tensors="pt",
                                  max_length=hparams.max_length) #SUPPORTS two PAIRs for now... !Tokenize inputs as a dict type of Tensors
#     print(inputs)

    
#     ds = NERSequenceDataset(inputs)
#     print(ds)
    
