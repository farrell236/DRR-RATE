import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from cxr_dataset import CheXpert
from models import CheXNet


def main():

    torch.multiprocessing.freeze_support()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(512, scale=(0.8, 1.0), ratio=(1., 1.)),
        # transforms.RandomHorizontalFlip(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    data_root = '/home/benjamin/mnt/cadlabnas/datasets/CheXpert_small_11GB/CheXpert-v1.0-small'
    dataset_train = CheXpert(data_root, split='train', transform=transform)
    dataset_valid = CheXpert(data_root, split='valid', transform=transform)

    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=4)
    dataloader_valid = DataLoader(dataset_valid, batch_size=32, num_workers=4)

    model = CheXNet(class_count=14, pretrained=False)
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints/chexpert-2', save_top_k=5, monitor='val_loss')
    trainer = pl.Trainer(max_epochs=100, accelerator='gpu', devices=1, callbacks=[checkpoint_callback])

    trainer.fit(model, dataloader_train, dataloader_valid)


if __name__ == '__main__':
    main()
