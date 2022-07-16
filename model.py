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
from dataset import SequenceDataset

#https://github.com/HelloJocelynLu/t5chem/blob/main/t5chem/archived/MultiTask.py for more info

# classifier.bert.pooler.dense.weight.requires_grad
# classifier.bert.pooler.dense.training
# classifier.classifier.weight.requires_grad
logging.basicConfig()
logger = logging.getLogger("BERT Fine-tuning")
logger.setLevel(logging.DEBUG)

class ProtBertClassifier(pl.LightningModule):
    """
    # https://github.com/minimalist-nlp/lightning-text-classification.git
    
    Sample model to show how to use BERT to classify sentences.
    
    :param hparams: ArgumentParser containing the hyperparameters.
    """

    def __init__(self, hparam: argparse.ArgumentParser) -> None:
        super(ProtBertClassifier, self).__init__()
        self.hparam = hparam
        self.ner_config = self.hparam.ner_config
        self.batch_size = self.hparam.batch_size
        self.model_name = self.hparam.model_name #"Rostlab/prot_bert_bfd"  
        self.ner = self.hparam.ner #bool
#         self.dataset = SequenceDataset.load_saved_dataset("data_compiled.pickle")
        self.num_labels = 9 #WIP: placeholder
        self.z_dim = self.hparam.z_dim #Add this!
        if self.hparam.loss == "contrastive": self.register_parameter("W", torch.nn.Parameter(torch.rand(self.z_dim, self.z_dim))) #CURL purpose
        
        # build model
        _ = self.__build_model() if not self.ner else self.__build_model_ner()

        # Loss criterion initialization.
        _ = self.__build_loss() if not self.ner else self.__build_loss_ner()

        self.freeze_encoder()

    def __build_model(self) -> None:
        """ Init BERT model + tokenizer + classification head."""
        #model = locals()["model"] if locals()["model"] and isinstance(locals()["model"], BertModel) else BertModel.from_pretrained(self.model_name, cache_dir=self.hparam.load_model_directory)
        model = BertModel.from_pretrained(self.model_name, cache_dir=self.hparam.load_model_directory)
 
        self.model = model
        self.encoder_features = self.z_dim

        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name, do_lower_case=False, return_tensors="pt", cache_dir=self.hparam.load_model_directory)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(self.encoder_features*4, self.encoder_features),
            nn.Linear(self.encoder_features, self.num_labels),
            nn.Tanh(),
        )
        if self.hparam.loss == "contrastive": 
            self.make_hook()

        wandb.init(project="DL_Sequence_Collab", entity="hyunp2", group="DDP_runs")
        wandb.watch(self.head)

    def __build_model_ner(self) -> None:
        """ Init BERT model + tokenizer + classification head.
        Model and Tokenizer has to be rewritten! 
        WIP!"""
        #model = locals()["model"] if locals()["model"] and isinstance(locals()["model"], BertModel) else BertModel.from_pretrained(self.model_name, cache_dir=self.hparam.load_model_directory)
        model = BertModel.from_pretrained(self.model_name, cache_dir=self.hparam.load_model_directory)
        
        self.model = model
        self.encoder_features = self.z_dim

        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name, do_lower_case=False, return_tensors="pt", cache_dir=self.hparam.load_model_directory)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(self.encoder_features, self.encoder_features),
            nn.Linear(self.encoder_features, self.num_labels)
        )
        if self.hparam.loss == "contrastive": 
            self.make_hook()

        wandb.init(project="DL_Sequence_Collab_Matt", entity="hyunp2")
        wandb.watch(self.head)
   
    def make_hook(self, ):
        self.fhook = dict()
        def hook(m, i, o):
            self.fhook["encoded_feats"] = o #(B,1024)
        self.fhook_handle = self.head[1].register_forward_hook(hook) #Call Forward hook with "model.fhook["encoded_feats"]" of (B,C); for NER, it is (B,L,C)
 
    def __build_loss(self):
        """ Initializes the loss function/s. """
        self._loss = nn.CrossEntropyLoss(label_smoothing=self.hparam.label_smoothing)

    def __build_loss_ner(self):
        """ Initializes the loss function/s. """
#         self._loss = CRF(num_tags=self.num_labels, batch_first=True)
        self._loss = torch.nn.BCEWithLogitsLoss(reduction="none")

    def compute_logits_CURL(self, z_a, z_pos):
        """
        WIP!
        https://github.com/MishaLaskin/curl/blob/8416d6e3869e38ca0e46fcbc54a2f784dc09d7fc/curl_sac.py#:~:text=def%20compute_logits(self,return%20logits
        Uses logits trick for CURL:
        - z_a and z_pos are last layer of classifier! z_pos is T(z_a) where T is Transformation func.
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        assert z_a.size(-1) == z_pos.size(-1) and z_a.size(-1) == self.z_dim, "dimension for CURL mismatch!"
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None] #(B,B)
        labels = torch.arange(logits.shape[0]).to(self.device).long() #(B,)
        loss = self._loss(logits, labels) #Use the defined CE
        return loss

    def unfreeze_encoder(self) -> None:
        """ un-freezes the encoder layer. """
        if self._frozen:
            logger.info(f"\n-- Encoder model fine-tuning")
            for param in self.model.parameters():
                param.requires_grad = True
            self._frozen = False

    def freeze_encoder(self) -> None:
        """ freezes the encoder layer. """
        for param in self.model.parameters():
            param.requires_grad = False
        self._frozen = True

    # def predict(self, sample: dict) -> dict:
    #     """ Predict function.
    #     :param sample: dictionary with the text we want to classify.
    #     Returns:
    #         Dictionary with the input text and the predicted label.
    #     """
    #     if self.training:
    #         self.eval()

    #     with torch.no_grad():
    #         model_input, _ = self.prepare_sample([sample], prepare_target=False)
    #         model_out = self.forward(**model_input)
    #         logits = model_out["logits"].numpy()
    #         predicted_labels = [
    #             self.label_encoder.index_to_token[prediction]
    #             for prediction in np.argmax(logits, axis=1)
    #         ]
    #         sample["predicted_label"] = predicted_labels[0]

    #     return sample
    
    # https://github.com/UKPLab/sentence-transformers/blob/eb39d0199508149b9d32c1677ee9953a84757ae4/sentence_transformers/models/Pooling.py
    @staticmethod
    def pool_strategy(features,
                      pool_cls=True, pool_max=True, pool_mean=True,
                      pool_mean_sqrt=True):
        token_embeddings = features['token_embeddings']
        cls_token = features['cls_token_embeddings']
        attention_mask = features['attention_mask']

        ## Pooling strategy
        output_vectors = []
        if pool_cls:
            output_vectors.append(cls_token)
        if pool_max:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if pool_mean or pool_mean_sqrt:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            #If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if 'token_weights_sum' in features:
                sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if pool_mean:
                output_vectors.append(sum_embeddings / sum_mask)
            if pool_mean_sqrt:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

        output_vector = torch.cat(output_vectors, 1)
        return output_vector #B,1024*4
    
    def forward(self, input_ids, token_type_ids, attention_mask, return_dict=True):    
        result = self.forward_classify(input_ids, token_type_ids, attention_mask, return_dict=True) if not self.ner else self.forward_ner(input_ids, token_type_ids, attention_mask, return_dict=True)
        return result #(B,2) or (BLC)

    def forward_classify(self, input_ids, token_type_ids, attention_mask, return_dict=True):
        """ Usual pytorch forward function.
        :param tokens: text sequences [batch_size x src_seq_len]
        :param lengths: source lengths [batch_size]
        Returns:
            Dictionary with model outputs (e.g: logits)
        """
        self.return_dict = return_dict
        
        word_embeddings = self.model(input_ids=input_ids, token_type_ids=token_type_ids,
                                           attention_mask=attention_mask)[0] #last_hidden_state

        pooling = self.pool_strategy({"token_embeddings": word_embeddings,
                                      "cls_token_embeddings": word_embeddings[:, 0],
                                      "attention_mask": attention_mask,
                                      }) 
        logits = self.head(pooling) #B2
        if return_dict:
            if self.hparam.loss == "classification":
                return {"logits": logits} #B,num_labels
            elif self.hparam.loss == "contrastive":
                logits = self.fhook["encoded_feats"]
                return {"logits": logits} #B, z_dim (differentiable)
        else:
            if self.hparam.loss == "classification":
                return logits #B,num_labels
            elif self.hparam.loss == "contrastive":
                logits = self.fhook["encoded_feats"]
                return logits #B, z_dim (differentiable)

    def forward_ner(self, input_ids, token_type_ids, attention_mask, return_dict=True):
        """ Usual pytorch forward function.
        :param tokens: text sequences [batch_size x src_seq_len]
        :param lengths: source lengths [batch_size]
        Returns:
            Dictionary with model outputs (e.g: logits)
        """
        self.return_dict = return_dict
        word_embeddings = self.model(input_ids=input_ids, token_type_ids=token_type_ids,
                                           attention_mask=attention_mask)[0] #last_hidden_state (BLC)
        logits = self.head(word_embeddings) #BLC
        
        if return_dict:
            if self.hparam.loss == "classification":
                return {"logits": logits} #BLC
        else:
            if self.hparam.loss == "classification":
                return logits #BLC

    def select_nonspecial(self, predictions: dict, inputs: dict, targets: dict):
        logits = predictions["logits"] if self.return_dict else predictions #dict or tensor!
        attention_mask = inputs["attention_mask"]
        labels = targets["labels"]
        target_invalid_lipids = targets["target_invalid_lipids"].view(-1,1,self.num_labels).expand_as(labels) #B,C -> B,L,C

        print(attention_mask.size(), logits.size(), labels.size())
        assert logits.size()[:2] == attention_mask.size() and logits.size(0) == labels.size(0), "logits and attention mask and labels must have the same dimension for non-channels/batch, each" #B,L+2/3
        attention_mask = (attention_mask.view(-1,) >= 5).expand(-1, self._num_labels) #(BL,) -> (BL,C);; type: torch.bool; choosing only non-special tokens!
        #ABOVE: https://gist.github.com/f1recracker/0f564fd48f15a58f4b92b3eb3879149b#:~:text=target%20%3D%20target%20*%20(target%20!%3D%20self.ignore_index).long()
        target_invalid_lipids: torch.ByteTensor = target_invalid_lipids.view(-1, self.num_labels) #(BL,C)
        tmp_stack = torch.stack([attention_mask, target_invalid_lipids], dim=-1) #(BL,C,2)
        boolean_tensor = tmp_stack.all(dim=-1).view(-1,) #(BL,C) -> (BLC,)
        
        predictions = predictions.view(-1,)[boolean_tensor] #(BLC,) -> (num_trues)
        labels = labels.view(-1,)[boolean_tensor] #(BLC,) -> (num_trues)
        return predictions, labels
        
    def loss(self, predictions: dict, targets: dict) -> torch.tensor:
        """
        Computes Loss value according to a loss function.
        :param predictions: model specific output. Must contain a key 'logits' with
            a tensor [batch_size x 1] with model predictions
        :param labels: Label values [batch_size]
        Returns:
            torch.tensor with loss value.
        """
        if self.hparam.loss == "classification" and not self.ner:
            return self._loss(predictions["logits"], targets["labels"].view(-1, )) #Crossentropy ;; input: (B,2) target (B,)
        elif self.hparam.loss == "classification" and self.ner:
#             return self._loss(predictions["logits"], targets["labels"].view(-1, self.num_labels)) #CRF ;; input (B,L,C) target (B,L) ;; B->num_frames & L->num_aa_residues & C->num_lipid_types
            return self._loss(predictions, targets.float()) #CRF ;; input (num_trues,C) target (num_trues,C) ;; 
        elif self.hparam.loss == "contrastive":
            return self.compute_logits_CURL(predictions["logits"], predictions["logits"]) #Crossentropy -> Need second pred to be transformed! each pred is (B,z_dim) shape

    def on_train_epoch_start(self, ) -> None:
        self.metric_acc = torchmetrics.Accuracy()
        #self.loss_log_for_train = []

    def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ 
        Runs one training step. This usually consists in the forward function followed
            by the loss function.
        
        :param batch: The output of your dataloader. 
        :param batch_nb: Integer displaying which batch this is
        Returns:
            - dictionary containing the loss and the metrics to be added to the lightning logger.
        """
        #import pdb; pdb.set_trace()
        inputs, targets = batch #Both are tesors
        #print(inputs, targets, "VALUE!!")
        model_out = self.forward(**inputs) #logicts dictionary
        model_out, targets = self.select_nonspecial(model_out, inputs, targets)
        loss_train = self.loss(model_out, targets) #BLC
#         loss_train = loss_train * targets["target_invalid_lipids"][:,None,:]
        
        y = targets["labels"].view(-1,)
        y_hat = model_out["logits"]
        labels_hat = torch.argmax(y_hat, dim=-1).to(y)
        train_acc = self.metric_acc(labels_hat.detach().cpu().view(-1,), y.detach().cpu().view(-1,)) #Must mount tensors to CPU;;;; ALSO, val_acc should be returned!

        output = {"train_loss": loss_train, "train_acc": train_acc} #NEVER USE ORDEREDDICT!!!!
        wandb.log(output)
        self.log("train_loss", loss_train, prog_bar=True)
        self.log("train_acc", train_acc, prog_bar=True)
        
        #self.loss_log_for_train.append({"train_acc": train_acc})

        return {"loss": loss_train, "train_acc": train_acc}

    def training_epoch_end(self, outputs: list) -> dict:
        """ Function that takes as input a list of dictionaries returned by the validation_step
        function and measures the model performance accross the entire validation set.
        
        Returns:
            - Dictionary with metrics to be added to the lightning logger.  
        """
        #outputs = self.loss_log_for_train
        train_loss_mean = torch.stack([x['train_acc'] for x in outputs]).mean()
#         train_acc_mean = self.metric_acc.compute()

        self.log("train_loss_mean", train_loss_mean, prog_bar=True)
        self.log("epoch", self.current_epoch)

#         self.log("train_acc_mean", train_acc_mean, prog_bar=True)

#         tqdm_dict = {"epoch_train_loss": train_loss_mean, "epoch_train_acc": train_acc_mean}
        tqdm_dict = {"epoch_train_loss": train_loss_mean}
        wandb.log(tqdm_dict)
        self.metric_acc.reset()    

    def on_validation_epoch_start(self, ) -> None:
        self.metric_acc = torchmetrics.Accuracy()

    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ Similar to the training step but with the model in eval mode.
        Returns:
            - dictionary passed to the validation_end function.
        #NEVER USE ORDEREDDICT!!!!
        """
        inputs, targets = batch
        model_out = self.forward(**inputs)
        #print(model_out.size(), targets["labels"].size())
        loss_val = self.loss(model_out, targets)
        y = targets["labels"].view(-1,) #B or BL
        y_hat = model_out["logits"] #B2 or BLC
        labels_hat = torch.argmax(y_hat, dim=-1).to(y)

        #print(y, labels_hat)
        val_acc = self.metric_acc(labels_hat.detach().cpu().view(-1), y.detach().cpu().view(-1)) #Must mount tensors to CPU;;;; ALSO, val_acc should be returned!

        output = {"val_loss": loss_val, "val_acc": val_acc} #NEVER USE ORDEREDDICT!!!!
        wandb.log(output)

        return output
        
    def validation_epoch_end(self, outputs: list) -> dict:
        """ Function that takes as input a list of dictionaries returned by the validation_step
        function and measures the model performance accross the entire validation set.
        
        Returns:
            - Dictionary with metrics to be added to the lightning logger.  
        """
        if not self.trainer.running_sanity_check:
            val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
            # val_acc_mean = torch.stack([x['val_acc'] for x in outputs]).mean()
#             val_acc_mean = self.metric_acc.compute()
 
            self.log("val_loss_mean", val_loss_mean, prog_bar=True)
#             self.log("val_acc_mean", torch.tensor(val_acc_mean).to(val_loss_mean), prog_bar=True) #tensor mount for Callback Early stop...https://pytorch-lightning.readthedocs.io/en/stable/_modules/pytorch_lightning/callbacks/early_stopping.html#:~:text=def%20_evaluate_stopping_criteria(self%2C%20current%3A%20torch.Tensor)%20%2D%3E%20Tuple%5Bbool%2C%20Optional%5Bstr%5D%5D%3A
            self.log("epoch", self.current_epoch, prog_bar=True)

#             tqdm_dict = {"epoch_val_loss": val_loss_mean, "epoch_val_acc": val_acc_mean}
            tqdm_dict = {"epoch_val_loss": val_loss_mean}

            wandb.log(tqdm_dict)
            self.metric_acc.reset()   

    def on_test_epoch_start(self, ) -> None:
        self.metric_acc = torchmetrics.Accuracy()

    def test_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ Similar to the training step but with the model in eval mode.
        Returns:
            - dictionary passed to the validation_end function.
        """
        inputs, targets = batch
        model_out = self.forward(**inputs)
        loss_test = self.loss(model_out, targets)
        y = targets["labels"].view(-1,)
        y_hat = model_out["logits"]
        labels_hat = torch.argmax(y_hat, dim=-1).to(y)
        #import pdb; pdb.set_trace()

        test_acc = self.metric_acc(labels_hat.detach().cpu().view(-1,), y.detach().cpu().view(-1)) #Must mount tensors to CPU
        
        output = {"test_loss": loss_test, "test_acc": test_acc}
        wandb.log(output)
        self.log("test_acc", test_acc, prog_bar=True)

        return output

    def test_epoch_end(self, outputs: list) -> dict:
        """ Function that takes as input a list of dictionaries returned by the validation_step
        function and measures the model performance accross the entire validation set.
        
        Returns:
            - Dictionary with metrics to be added to the lightning logger.  
        """
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
#         test_acc_mean = self.metric_acc.compute()
#         tqdm_dict = {"epoch_test_loss": test_loss_mean, "epoch_test_acc": test_acc_mean}
        tqdm_dict = {"epoch_test_loss": test_loss_mean}

        wandb.log(tqdm_dict)
        self.metric_acc.reset()   

    def on_predict_epoch_start(self, ):
        if self.hparam.loss == "classification":
            self.make_hook() #Get a hook if classification was originally trained
        if self.hparam.loss == "ner":
            self.make_hook() #Get a hook if ner was originally trained
        elif self.hparam.loss == "contrastive": 
            pass

    def predict_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ Similar to the training step but with the model in eval mode.
        Returns:
            - dictionary passed to the validation_end function.
        """
        inputs, targets = batch
        model_out = self.forward(**inputs)
        loss_test = self.loss(model_out, targets)
        ground_truth = targets["labels"].view(-1,) #B or (BL)
        y_hat = model_out["logits"]
        predictions = y_hat.view(-1, self.num_labels) #B,2 or (BL,C)

        logits = self.fhook["encoded_feats"] #(B,z_dim) or (B,L,z_dim)
        #assert logits.requires_grad, "it is not differentiable"
        #import pdb; pdb.set_trace()
        
        return {"ground_truth": ground_truth, "predictions": predictions, "last_layer": logits}

    def on_predict_epoch_end(self, outputs: list) -> dict:
        """ Function that takes as input a list of dictionaries returned by the validation_step
        function and measures the model performance accross the entire validation set.
        
        Returns:
            - Dictionary with metrics to be added to the lightning logger.  
        """
        ground_truth = torch.cat([x['ground_truth'] for x in outputs[0]], dim=0).contiguous().view(-1,).detach().cpu().numpy() #catted results (B,) or (BL, )
        b = ground_truth.shape[0]
        predictions = torch.cat([x['predictions'] for x in outputs[0]], dim=0).contiguous().view(b,-1).detach().cpu().numpy() #catted results ... weird outputs indexing (B, num_labels) or (BL, num_labels)
        predictions = predictions.argmax(axis = -1) #(B,) or (BL,)
        class_names = np.arange(self.num_labels).tolist() #(num_labels)
        logits = torch.cat([x['last_layer'] for x in outputs[0]], dim=0).contiguous().view(b,-1).detach().cpu().numpy() #catted results (B,reduced_dim)
        logits_ = logits #(B,zdim) or (B,L,zdim)
        if self.ner: logits_ = logits_[:,0,:] #to (B,zdim) from [CLS]
        
        self.plot_confusion(ground_truth, predictions, class_names)
        self.plot_manifold(self.hparam, logits_) #WIP to have residue projection as well!
        self.plot_ngl(self.hparam)
 
    @staticmethod
    def plot_confusion(ground_truth: torch.Tensor, predictions: torch.Tensor, class_names: np.ndarray=np.array([0,1])):
        cm = wandb.plot.confusion_matrix(
            y_true=ground_truth,
            preds=predictions,
            class_names=class_names) 
        wandb.log({"Confusion Matrix": cm}) #Needs (B, )
        wandb.log({"ROC": wandb.plot.roc_curve(ground_truth, predictions)}) #Needs (B/BL,num_labels)
        wandb.log({"PR": wandb.plot.pr_curve(ground_truth, predictions)}) #Needs (B/BL,num_labels)
  
    @staticmethod
    def plot_manifold(hparam: argparse.ArgumentParser, logits_: np.ndarray):
        #WIP for PCA or UMAP or MDS
        #summary is 
        import sklearn.manifold
        import plotly.express as px
        tsne = sklearn.manifold.TSNE(2)
        logits_tsne = tsne.fit_transform(logits_) #(B,2) of tsne
        path_to_plotly_html = os.path.join(hparam.load_model_directory, "plotly_figure.html")
        fig = px.scatter(x=logits_tsne[:,0], y=logits_tsne[:,1], color=ground_truth)
        fig.write_html(path_to_plotly_html, auto_play = False)
        table = wandb.Table(columns = ["plotly_figure"])
        table.add_data(wandb.Html(path_to_plotly_html))
        wandb.log({"TSNE Plot": table})

    @staticmethod
    def plot_ngl(hparam: argparse.ArgumentParser):
        #WIP for filename!
        import nglview as nv
        import MDAnalysis as mda
        universe = mda.Universe("/Scr/hyunpark/ZIKV_ConcatDCD_for_DL/REDO/data/alanine-dipeptide-nowater_charmmgui.psf", "/Scr/hyunpark/ZIKV_ConcatDCD_for_DL/REDO/data/alanine-dipeptide-nowater_charmmgui.pdb")   
        u = universe.select_atoms("all")
        w = nv.show_mdanalysis(u)
        w.clear_representations()
        #w.add_licorice(selection="protein")
        w.add_representation('licorice', selection='all', color='blue')
        w.render_image(factor=1, frame=3, transparent=False)
        path_to_ngl_html = os.path.join(hparam.load_model_directory, "nglview_figure.html")
        nv.write_html(path_to_ngl_html, [w])     
        table = wandb.Table(columns = ["nglview_figure"])
        table.add_data(wandb.Html(path_to_ngl_html))
        wandb.log({"NGL View": table}) 

    def configure_optimizers(self):
        """ Sets different Learning rates for different parameter groups. """
        parameters = [
            {"params": self.head.parameters()},
            {"params": self.model.parameters()},
            ]
        if self.hparam.optimizer == "adafactor":
            optimizer = Adafactor(parameters, relative_step=True)
        elif self.hparam.optimizer == "adamw":
            optimizer = torch.optim.AdamW(parameters, lr=self.hparam.learning_rate)
        total_training_steps = len(self.train_dataloader()) * self.hparam.max_epochs
        warmup_steps = total_training_steps // self.hparam.warm_up_split
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )
#         optimizer = {"optimizer": optimizer, "frequency": 1}
        #https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#:~:text=is%20shown%20below.-,lr_scheduler_config,-%3D%20%7B%0A%20%20%20%20%23%20REQUIRED
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1} #Every step/epoch with Frequency 1etc by monitoring val_loss if needed

        return [optimizer], [scheduler]
    
    @staticmethod
    def _get_split_sizes(train_frac: float, full_dataset: torch.utils.data.Dataset) -> Tuple[int, int, int]:
        """DONE: Need to change split schemes!"""
        len_full = len(full_dataset)
        len_train = int(len_full * train_frac)
        len_test = int(0.1 * len_full)
        len_val = len_full - len_train - len_test
        return len_train, len_val, len_test  
    
    def load_dataset(self, stage="train"):
        if self.hparam.save_to_file != None:
            if os.path.exists(self.hparam.save_to_file):
                filename = os.path.splitext(self.hparam.save_to_file)[0] + ".pickle"
                dataset = dl.SequenceDataset.load_saved_dataset(filename)
            else:
                dataset = dl.SequenceDataset.from_directory(self.hparam.json_directory, self.hparam)
        else:
            dataset = dl.SequenceDataset.from_directory(self.hparam.json_directory, self.hparam)
        train, val, test = torch.utils.data.random_split(dataset, self._get_split_sizes(self.hparam.train_frac, dataset),
                                                                generator=torch.Generator().manual_seed(0))
        
        if stage == "train":
            dataset = train
        elif stage == "val":
            dataset = val
        elif stage == "test":
            dataset = val
        
        return dataset #torch Dataset

    def train_dataloader(self) -> DataLoader:
        """ Function that loads the train set. """
        self._train_dataset = self.load_dataset(stage="train")
        return DataLoader(
            dataset=self._train_dataset,
            sampler=torch.utils.data.RandomSampler(self._train_dataset),
            batch_size=self.hparam.batch_size,
            num_workers=self.hparam.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        self._dev_dataset = self.load_dataset(stage="test")
        return DataLoader(
            dataset=self._dev_dataset,
            batch_size=self.hparam.batch_size,
            num_workers=self.hparam.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        self._test_dataset = self.load_dataset(stage="test")
        return DataLoader(
            dataset=self._test_dataset,
            batch_size=self.hparam.batch_size,
            num_workers=self.hparam.num_workers,
        )

if __name__ == "__main__":
    from train import get_args
    hparams = get_args()
    ds = SequenceDataset.from_directory("/Scr/hyunpark/DL_Sequence_Collab/ProtLIpInt", hparams) #concat dataset instance
    dl = torch.utils.data.DataLoader(ds, batch_size=15)   
    one_ds = iter(dl).next()
    inputs, targets = one_ds
    
    model = ProtBertClassifier(hparams)
    outs = model(**inputs)
    print(outs)
    out, tar = model.select_nonspecial(outs, inputs, targets)
    print(out,tar)
    loss = model.loss(out,tar)
    print(loss)
    

    
