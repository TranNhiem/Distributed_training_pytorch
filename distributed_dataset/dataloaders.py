'''
@TranNhiem 2022.
we use a PyTorch Lightning Data Module designed to load one partition per epoch. 
This module represents the feature and label records from dataloaders.py as a PyTorch dataset.
At each new training epoch, it will load a new data partition, and, under-the hood, wrap a PyTorch dataloader in a distributed sampler, 
--> so that each record from the data partition is used only once. 

'''

import os
import torch
import pickle

from Flag_configs.multi_gpus_training_config import read_cfg
from Flag_configs.absl_mock import Mock_Flag

import pytorch_lightning as pl
from typing import Optional
from torch.utils.data import Dataset, DataLoader


read_cfg()
flag = Mock_Flag()
FLAGS = flag.FLAGS

class PartitionDataset(Dataset):
    
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data["features"])
    def __getitem__(self, idx):
        return (
            self.data["features"][idx],
            self.data["labels"][idx]
        )

class dataloaders_module(pl.LightningDataModule):

    def __init__(
        self, batch_size, train_files, val_file, num_workers=2
    ):
        super().__init__()
        self.train_files = sorted(train_files)
        self.val_file = val_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_data = self.load_data(self.val_file)
    
    def load_data(self, file):
        '''
        Helper function to load dataset with existing Pickle File
        
        '''
       
        with open(file, "rb") as f:
            data = pickle.load(f)
        return data


    def prepare_data(self):
        """
         prepare and loading dataset from the Disk in or Downloading dataset with Existing URL

        """
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Anything called here is being distributed across GPUs
        (do many times).  Lightning handles distributed sampling.
        """
        # Build the val dataset
        self.val_dataset = PartitionDataset(data=self.val_data)
    
    def train_dataloader(self):
        """
        This function sends the same file to each GPU and
        loops back after running out of files.
        Lightning will apply distributed sampling to
        the data loader so that each GPU receives
        different samples from the file until exhausted.
        """
        # Load the data file with the right index
        total = len(self.train_files)        
        train_file_idx = self.trainer.current_epoch % total
        train_file = self.train_files[train_file_idx]
        train_data = self.load_data(train_file)
        # Build the train dataset
        train_dataset = PartitionDataset(data=train_data)
        # Return the dataloader, which lightning will turn
        # into a distributed data loader, ensuring that
        # different samples are selected on each GPU
        return DataLoader(
            train_dataset,
            self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )