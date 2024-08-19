import torch
import pytorch_lightning as pl
from torch import nn


class Model:
    pass


class LitModel(pl.LightningModule):
    def __init__(self, model, training_params):
        super(LitModel, self).__init__()
        self.model = model
        self.learning_rate = training_params.get('learning_rate', 0.001)
        self.weight_decay = training_params.get('weight_decay', 0.0)
        self.momentum = training_params.get('momentum', 0.9)
        self.optimizer_type = training_params.get('optimizer', 'adam')
        
        # Determine the loss function based on training_params
        loss_function = training_params.get('loss_function', 'mse').lower()
        if loss_function == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_function == 'mae':
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        if self.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")
        
        return optimizer