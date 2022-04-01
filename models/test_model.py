import torch
import pytorch_lightning as pl
import torch.nn.functional as F 
import math
from torch import nn



class 

class test_model(pl.LightningModule): 
    def __init__(self, num_layer, num_classes): 
        super().__init__()
        self.num_layer= num_classes
        self.num_classes= num_classes

    def training_step(self, batch, batch_idx):
        data, labels = batch
        yhat = self.forward(data)
        train_loss = F.binary_cross_entropy_with_logits(
            yhat, labels
        )
        self.log("train_loss", train_loss, on_epoch=True)
        return train_loss
    def validation_step(self, batch, batch_idx):
        data, labels = batch
        yhat = self.forward(data)
        val_loss = F.binary_cross_entropy_with_logits(
            yhat, labels
        )
        self.log("val_loss", val_loss, on_epoch=True)
    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)