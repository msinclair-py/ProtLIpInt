# ProtLIpInt
## preprocessing.py
### This is a library of functions to preprocess pdb data for the
### Deep Learning algorithm.

## training_data.py
### This is a script for obtaining the lipid binding profile of
### each protein resID from a trajectory or suite of trajectories.

## obtain_training_data.ipynb
### This a jupyter notebook for testing the training_data.py script

## train.py
### This is the main ML training script. Callbacks and Trainer are defined.

## dataset.py
### This is a hugginging face dataset (and more; WIP) and Pytorch Dataset script.

## model.py
### Ths is a Pytorch Lightning Module and Datamodule script.

## utils.py
### This is a script for prediction, visualization and more.

## WIP: To train the model, 
<code><br>CUDA_VISIBLE_DEVICES=0 python train.py -ls 0.1 -b 512 -ckpt epoch=4-val_loss=0.30-val_acc=0.94.ckpt</code></br> 
