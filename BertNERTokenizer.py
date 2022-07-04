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
from SmilesPE.pretokenizer import atomwise_tokenizer, kmer_tokenizer
import inspect
import functools
import json
import itertools

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
    r"""
    Constructs a SMILES tokenizer. Based on SMILES Pair Encoding (https://github.com/XinhaoLi74/SmilesPE).
    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the methods. Users
    should refer to the superclass for more information regarding methods.
    Args:
        vocab_file (:obj:`string`):
            File containing the vocabulary.
        spe_file (:obj:`string`):
            File containing the trained SMILES Pair Encoding vocabulary.
        unk_token (:obj:`string`, `optional`, defaults to "[UNK]"):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (:obj:`string`, `optional`, defaults to "[SEP]"):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences
            for sequence classification or for a text and a question for question answering.
            It is also used as the last token of a sequence built with special tokens.
        pad_token (:obj:`string`, `optional`, defaults to "[PAD]"):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (:obj:`string`, `optional`, defaults to "[CLS]"):
            The classifier token which is used when doing sequence classification (classification of the whole
            sequence instead of per-token classification). It is the first token of the sequence when built with
            special tokens.
        mask_token (:obj:`string`, `optional`, defaults to "[MASK]"):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
    """

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
    
    seq = list(data.keys()) #e.g. TYR-483-PROA
    seq_nonzero = list(map(lambda s: list(filter(lambda l: l, data[s].values())), seq)) #List[List of COEFF]
    split_txt = np.array(list(map(lambda inp: inp.split("-"), seq))) #List[tuple of RESNAME_RESID_SEGID] -> np.array
    
    ##AA Letter Mapping
    AA = ["ALA","ARG","ASN","ASP","CYS","GLU","GLN","GLY","HIS","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"]
    aa = ["A","R","N","D","C","E","Q","H","I","L","K","M","F","P","S","T","W","Y","V"]
    three2one = {THREE:ONE for THREE, ONE in list(zip(AA,aa))}
    seqs = list(map(lambda seq: ' '.join(list(map(lambda aa: three2one.get(aa, None), seq.split(" ") ))), seqs ))    #THREE LETTER -> ONE LETTER
    
    ##Lipid index dictionary
    lips = ["PC","PE","PG","PI","PS","PA","CL","SM","CHOL","OTHERS"] 
    lip2idx = {k:v for v, k in enumerate(lips)}
    idx2lip = {v:k for k, v in lip2idx.items()}
    
    segs = ["PROA","PROB","PROC","PROD"] #use [SEP] for different segment!
#     aa_seg = list(itertools.product(aa, seg))

    #DATA PREPROCESSING
    all_resnames, all_segnames, modified_slice = split_txt[:,0].tolist(), split_txt[:,2], []
    all_resnames = list(map(lambda seq: ' '.join(list(map(lambda aa: three2one.get(aa, None), seq.split(" ") ))), all_resnames ))
    start_idx = 0
    for seg in segs:
        end_idx_p1 = np.sum(all_segnames == seg) + start_idx
        current_slice = all_resnames[slice(start_idx, end_idx_p1)] + ["[SEP]"]
        modified_slice += current_slice
        start_idx = end_idx_p1
    modified_slice.pop() #last SEP token should be gone! #<SEQ1 + SEP + SEQ2 + SEP + SEQ3 ...>
    
    print(modified_slice)
    
#     # some default tokens from huggingface
#     # Manually collected
#     default_toks = ['[PAD]', 
#                     '[unused1]', '[unused2]', '[unused3]', '[unused4]','[unused5]', '[unused6]', '[unused7]', '[unused8]', '[unused9]', '[unused10]', 
#                     '[UNK]', '[CLS]', '[SEP]', '[MASK]']

#     # atom-level tokens used for trained the spe vocabulary
#     # This can be obtained from SmilesPE.learner (see tokenizers/examples.py)
#     atom_toks = ['[c-]', '[SeH]', '[N]', '[C@@]', '[Te]', '[OH+]', 'n', '[AsH]', '[B]', 'b', 
#                  '[S@@]', 'o', ')', '[NH+]', '[SH]', 'O', 'I', '[C@]', '-', '[As+]', '[Cl+2]', 
#                  '[P+]', '[o+]', '[C]', '[C@H]', '[CH2]', '\\', 'P', '[O-]', '[NH-]', '[S@@+]', 
#                  '[te]', '[s+]', 's', '[B-]', 'B', 'F', '=', '[te+]', '[H]', '[C@@H]', '[Na]', 
#                  '[Si]', '[CH2-]', '[S@+]', 'C', '[se+]', '[cH-]', '6', 'N', '[IH2]', '[As]', 
#                  '[Si@]', '[BH3-]', '[Se]', 'Br', '[C+]', '[I+3]', '[b-]', '[P@+]', '[SH2]', '[I+2]', 
#                  '%11', '[Ag-3]', '[O]', '9', 'c', '[N-]', '[BH-]', '4', '[N@+]', '[SiH]', '[Cl+3]', '#', 
#                  '(', '[O+]', '[S-]', '[Br+2]', '[nH]', '[N+]', '[n-]', '3', '[Se+]', '[P@@]', '[Zn]', '2', 
#                  '[NH2+]', '%10', '[SiH2]', '[nH+]', '[Si@@]', '[P@@+]', '/', '1', '[c+]', '[S@]', '[S+]', 
#                  '[SH+]', '[B@@-]', '8', '[B@-]', '[C-]', '7', '[P@]', '[se]', 'S', '[n+]', '[PH]', '[I+]', '5', 'p', '[BH2-]', '[N@@+]', '[CH]', 'Cl']

    
    
#     # spe tokens
#     # Made from SmilesPE.learner
#     # SPE format must be "[FEATURIZER]_[SPE]_[DATABASE].txt"
#     with open('tokenization/SMILES_SPE_ChEMBL.txt', "r") as ins:
#         spe_toks = []
#         for line in ins:
#             spe_toks.append(line.split('\n')[0])

#     spe_tokens = []
#     for s in spe_toks:
#         spe_tokens.append(''.join(s.split(' ')))
#     print('Number of SMILES:', len(spe_toks))
    
#     spe_vocab = default_toks + atom_toks + spe_tokens
    
#     #Make a vocabulary file
#     with open('tokenization/vocab_spe.txt', 'w') as f:
#         for voc in spe_vocab:
#             f.write(f'{voc}\n')
            
#     tokenizer = SMILES_SPE_Tokenizer(vocab_file='tokenization/vocab_spe.txt', spe_file='tokenization/SMILES_SPE_ChEMBL.txt', which_featurizer="smiles", which_tokenizer="spe")
    
#     smi_1 = 'CC[N+](C)(C)Cc1ccccc1Br'
#     smi_2 = 'c1cccc1[invalid]'
# #     encoded_input = tokenizer(smi_1,smi_2) #Do not use this;; similar to EmbeddingBag
# #     tokenizer.decode(encoded_input["input_ids"])  #Do not use this;; similar to EmbeddingBag
#     encoded_input = tokenizer.batch_encode_plus([smi_1,smi_2], 
#                                                   add_special_tokens=True,
#                                                   padding=True, truncation=True, return_tensors="pt",
#                                                   max_length=100) #USE this;; similar to Embedding
#     tokenizer.batch_decode(encoded_input["input_ids"]) #USE this;; similar to Embedding
