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
from dataset import SequenceDataset, NERSequenceDataset

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
    
# class BertNERTokenizer(PreTrainedTokenizer):
#     def __init__(
#         self,
#         vocab_file,
#         spe_file,
#         unk_token="[UNK]",
#         sep_token="[SEP]",
#         pad_token="[PAD]",
#         cls_token="[CLS]",
#         mask_token="[MASK]",
#         which_featurizer="smiles",
#         which_tokenizer="spe",
#         **kwargs
#     ):
#         super().__init__(
#             unk_token=unk_token,
#             sep_token=sep_token,
#             pad_token=pad_token,
#             cls_token=cls_token,
#             mask_token=mask_token,
#             **kwargs,
#         )
#         self.which_featurizer = which_featurizer
#         self.which_tokenizer = which_tokenizer
        
#         if not os.path.isfile(vocab_file):
#             raise ValueError(
#                 "Can't find a vocabulary file at path '{}'.".format(vocab_file)
#             )
#         if not os.path.isfile(spe_file):
#             raise ValueError(
#                 "Can't find a SPE vocabulary file at path '{}'.".format(spe_file)
#             )
#         self.vocab = load_vocab(vocab_file) #Covers both Atomwise and SPE; not KMER!
#         self.spe_vocab = codecs.open(spe_file)
#         self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        
#         if which_featurizer == "smiles":
#             assert which_featurizer in spe_file.lower(), "Vocabulary learning was not done in SMILES format..."
#         elif which_featurizer == "deepsmiles":
#             assert which_featurizer in spe_file.lower(), "Vocabulary learning was not done in DEEPSMILES format..."
#         elif which_tokenizer == "selfies":
#             assert which_featurizer in spe_file.lower(), "Vocabulary learning was not done in SELFIES format..."
#         else:
#             raise FeaturizerNotExistentError()        
        
#         if which_tokenizer == "spe":
#             self.tokenizer = SPE_Tokenizer_Wrapped(self.spe_vocab)
#         elif which_tokenizer == "atom":
#             self.tokenizer = atomwise_tokenizer
# #         elif which_tokenizer == "kmer":
# #             #WIP Do not use KMER yet...; hard to build vocabulary
# #             self.tokenizer = kmer_tokenizer
#         else:
#             raise TokenizerNotExistentError()

#     @property
#     def vocab_size(self):
#         return len(self.vocab)

#     def get_vocab(self):
#         return dict(self.vocab, **self.added_tokens_encoder)

#     def _tokenize(self, text):
#         return self.tokenizer(text).split(' ') if self.which_tokenizer == "spe" else self.tokenizer(text) #spe gives a string with spaces; others give a list

#     def _convert_token_to_id(self, token):
#         """ Converts a token (str) in an id using the vocab. """
#         return self.vocab.get(token, self.vocab.get(self.unk_token))

#     def _convert_id_to_token(self, index):
#         """Converts an index (integer) in a token (str) using the vocab."""
#         return self.ids_to_tokens.get(index, self.unk_token)

#     def convert_tokens_to_string(self, tokens):
#         """ Converts a sequence of tokens (string) in a single string. """
#         out_string = " ".join(tokens).replace(" ##", "").strip()
#         return out_string

#     def build_inputs_with_special_tokens(
#         self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, token_ids_2: Optional[List[int]] = None
#     ) -> List[int]:
#         """
#         Build model inputs from a sequence or a pair of sequence for sequence classification tasks
#         by concatenating and adding special tokens.
#         A BERT sequence has the following format:
#         - single sequence: ``[CLS] X [SEP]``
#         - pair of sequences: ``[CLS] A [SEP] B [SEP]``
#         Args:
#             token_ids_0 (:obj:`List[int]`):
#                 List of IDs to which the special tokens will be added
#             token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
#                 Optional second list of IDs for sequence pairs.
#         Returns:
#             :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
#         """
#         cls = [self.cls_token_id]
#         sep = [self.sep_token_id]

#         if token_ids_1 is None:
#             return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
#         elif token_ids_2 is None:
#             return cls + token_ids_0 + sep + token_ids_1 + sep
#         else:
#             return cls + token_ids_0 + sep + token_ids_1 + sep + token_ids_2 + sep
        
#     def get_special_tokens_mask(
#         self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, token_ids_2: Optional[List[int]] = None, already_has_special_tokens: bool = False
#     ) -> List[int]:
#         """
#         Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
#         special tokens using the tokenizer ``prepare_for_model`` method.
#         Args:
#             token_ids_0 (:obj:`List[int]`):
#                 List of ids.
#             token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
#                 Optional second list of IDs for sequence pairs.
#             already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
#                 Set to True if the token list is already formatted with special tokens for the model
#         Returns:
#             :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
#         """

#         if already_has_special_tokens:
#             if token_ids_1 is not None:
#                 raise ValueError(
#                     "You should not supply a second sequence if the provided sequence of "
#                     "ids is already formated with special tokens for the model."
#                 )
#             return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

#         if token_ids_1 is None:
#             return [1] + ([0] * len(token_ids_0)) + [1]
#         elif token_ids_2 is None:
#             return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
#         else:
#             return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1] + ([0] * len(token_ids_2)) + [1]
        
#     def create_token_type_ids_from_sequences(
#         self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, token_ids_2: Optional[List[int]] = None
#     ) -> List[int]:
#         """
#         Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
#         A BERT sequence pair mask has the following format:
#         ::
#             0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
#             | first sequence    | second sequence |
#         if token_ids_1 is None, only returns the first portion of the mask (0's).
#         Args:
#             token_ids_0 (:obj:`List[int]`):
#                 List of ids.
#             token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
#                 Optional second list of IDs for sequence pairs.
#         Returns:
#             :obj:`List[int]`: List of `token type IDs <../glossary.html#token-type-ids>`_ according to the given
#             sequence(s).
#         """
#         sep = [self.sep_token_id]
#         cls = [self.cls_token_id]
#         if token_ids_1 is None:
#             return len(cls + token_ids_0 + sep) * [0]
#         elif token_ids_2 is None:
#             return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]
#         else:
#             return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1] + len(token_ids_2 + sep) * [2]
        
#     def save_vocabulary(self, vocab_path):
#         """
#         Save the sentencepiece vocabulary (copy original file) and special tokens file to a directory.
#         Args:
#             vocab_path (:obj:`str`):
#                 The directory in which to save the vocabulary.
#         Returns:
#             :obj:`Tuple(str)`: Paths to the files saved.
#         """
#         index = 0
#         if os.path.isdir(vocab_path):
#             vocab_file = os.path.join(vocab_path, VOCAB_FILES_NAMES["vocab_file"])
#         else:
#             vocab_file = vocab_path
#         with open(vocab_file, "w", encoding="utf-8") as writer:
#             for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
#                 if index != token_index:
#                     logger.warning(
#                         "Saving vocabulary to {}: vocabulary indices are not consecutive."
#                         " Please check that the vocabulary is not corrupted!".format(vocab_file)
#                     )
#                     index = token_index
#                 writer.write(token + "\n")
#                 index += 1
#         return (vocab_file,)

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/tokenization_bert.py
from transformers.models.bert.tokenization_bert import *
from transformers.tokenization_utils import *
class BertTokenizer2(PreTrainedTokenizer):
    r"""
    Construct a BERT tokenizer. Based on WordPiece.
    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        do_basic_tokenize (`bool`, *optional*, defaults to `True`):
            Whether or not to do basic tokenization before WordPiece.
        never_split (`Iterable`, *optional*):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            `do_basic_tokenize=True`
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters.
            This should likely be deactivated for Japanese (see this
            [issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs
    ):
        super().__init__(
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)

    @property
    def do_lower_case(self):
        return self.basic_tokenizer.do_lower_case

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text):
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):

                # If the token is part of the never_split set
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    #https://huggingface.co/transformers/v3.0.2/_modules/transformers/tokenization_bert.html#BertTokenizer.build_inputs_with_special_tokens
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, token_ids_2: Optional[List[int]] = None
    ) -> List[int]:
        """Custom build"""
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]

        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        elif token_ids_2 is None:
            return cls + token_ids_0 + sep + token_ids_1 + sep
        else:
            return cls + token_ids_0 + sep + token_ids_1 + sep + token_ids_2 + sep
        
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, token_ids_2: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """Custom build"""
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        elif token_ids_2 is None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        else:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1] + ([0] * len(token_ids_2)) + [1]
        
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, token_ids_2: Optional[List[int]] = None
    ) -> List[int]:
        """Custom build"""
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        elif token_ids_2 is None:
            return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]
        else:
            return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1] + len(token_ids_2 + sep) * [2]
        
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)

if __name__ == "__main__":      
    with open("sample_output_coeffs.json", "r") as f:
        data = json.load(f)
    
    seq = list(data.keys()) #e.g. TYR-483-PROA
#     seq_nonzero = list(map(lambda s: list(filter(lambda l: l, data[s].values())), seq)) #List[List of COEFF]
    split_txt = np.array(list(map(lambda inp: inp.split("-"), seq))) #List[tuple of RESNAME_RESID_SEGID] -> np.array
    
    ##AA Letter Mapping
    AA = ["ALA","ARG","ASN","ASP","CYS","GLU","GLN","GLY","HSD","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"]
    aa = ["A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
    three2one = {THREE:ONE for THREE, ONE in list(zip(AA,aa))}
    seq_parser = lambda seqs: list(map(lambda seq: ' '.join(list(map(lambda aa: three2one.get(aa, None), seq.split(" ") ))), seqs ))    #THREE LETTER -> ONE LETTER
#     print(np.isin(split_txt[:,0], AA).all())

    duplicates = 10

    ##Lipid index dictionary
    lips = ["PC","PE","PG","PI","PS","PA","CL","SM","CHOL","OTHERS"] #This specific order matters!
    lip2idx = {k:v for v, k in enumerate(lips)}
    idx2lip = {v:k for k, v in lip2idx.items()}
    def lip_index(data: dict):
        seq = list(data.keys()) #e.g. TYR-483-PROA
#         seq_nonzero = list(map(lambda s: list(filter(lambda l: l, data[s].values())), seq)) #List[List of COEFF]
#         seq_nonzero = [[(lip2idx.get(l, None), v) for (l, v) in data[s].items() if isinstance(v, list)] for s in seq]
        seq_nonzero = [[[(lip2idx.get(l, None), v if isinstance(v, list) else [v]*3) for (l, v) in data[s].items()] for s in seq]]
        return seq_nonzero
    lip_data = lip_index(data) * duplicates
    print(len(lip_data[0])) #For 1 data, [num_AA lists; each AA list has 8 lipid type tuples];;; #172
    segs = ["PROA","PROB","PROC","PROD"] #use [SEP] for different segment!
#     aa_seg = list(itertools.product(aa, seg))

    #DATA PREPROCESSING
    split_txt = np.tile(split_txt, (2,1)) #Multiseg-test
    split_txt[len(seq):len(seq)*2,2] = "PROB" #Multiseg-test
    split_txt[2*len(seq):,2] = "PROC" #Multiseg-test
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
    proper_inputs = proper_inputs * duplicates #List[10 lists of sent pairs]
#     print(len(proper_inputs))
    print(modified_slice[0].__len__()) #172

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
    tokenizer=BertTokenizer2.from_pretrained("Rostlab/prot_bert",do_lower_case=False, return_tensors="pt",cache_dir=hparams.load_model_directory)
#     tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
#     tokenizer.get_special_tokens_mask = get_special_tokens_mask
#     tokenizer.create_token_type_ids_from_sequences = create_token_type_ids_from_sequences
    print(tokenizer.create_token_type_ids_from_sequences.__doc__)

    inputs = tokenizer.batch_encode_plus(proper_inputs,
                                  add_special_tokens=True,
                                  padding=True, truncation=True, return_tensors="pt",
                                  max_length=hparams.max_length) #SUPPORTS two PAIRs for now... !Tokenize inputs as a dict type of Tensors
    targets = lip_data
#     print(inputs)
    ds = SequenceDataset(inputs, targets)
#     print(ds)
    dl = torch.utils.data.DataLoader(ds)
    print(iter(dl).next())
#     ds = NERSequenceDataset(inputs)
#     print(ds)
    
    
    
    

        
