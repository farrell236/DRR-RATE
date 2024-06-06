import torch
from torch import nn
from torch.nn import functional as F

import torchvision
import pytorch_lightning as pl


class CheXNet(pl.LightningModule):
    def __init__(self, class_count=14, pretrained=False, chexnet_version=121):
        super(CheXNet, self).__init__()

        # Validate and set the DenseNet version
        if chexnet_version not in (121, 169, 201):
            raise ValueError("CheXNet must be one of 121, 169, or 201")

        # Dynamically select the DenseNet model
        if chexnet_version == 121:
            self.chexnet = torchvision.models.densenet121(pretrained=pretrained)
        elif chexnet_version == 169:
            self.chexnet = torchvision.models.densenet169(pretrained=pretrained)
        elif chexnet_version == 201:
            self.chexnet = torchvision.models.densenet201(pretrained=pretrained)

        # Retrieve the number of input features to the classifier
        kernel_count = self.chexnet.classifier.in_features

        # Replace the classifier with a new one suitable for the number of classes
        self.chexnet.classifier = nn.Sequential(
            nn.Linear(kernel_count, class_count),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.chexnet(x)

    def cross_entropy_loss(self, logits, labels):
      return F.binary_cross_entropy(logits.float(), labels.float())

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=1e-4, betas=(0.9, 0.999),
                                     eps=1e-08, weight_decay=1e-5)
        return optimizer