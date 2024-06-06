import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from sklearn.model_selection import KFold

from cxr_dataset import NIH_CXR
from models import CheXNet


k_folds = 5
batch_size = 32


def main():

    torch.multiprocessing.freeze_support()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(512, scale=(0.8, 1.0), ratio=(1., 1.)),
        # transforms.RandomHorizontalFlip(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    data_root = '/home/benjamin/mnt/cadlabnas/users/Benjamin/data/NIH-CXR'
    dataset_train_val = NIH_CXR(data_root, split='train_val', transform=transform)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    for idx, (train_idx, valid_idx) in enumerate(kf.split(dataset_train_val)):

        if idx in [0]: continue

        print(f"Training Fold {idx}... \n"
              f"TRAIN: {train_idx}, TEST: {valid_idx}")

        dataloader_train = DataLoader(dataset_train_val, batch_size=batch_size, num_workers=4,
                                      sampler=torch.utils.data.SubsetRandomSampler(train_idx))
        dataloader_valid = DataLoader(dataset_train_val, batch_size=batch_size, num_workers=4,
                                      sampler=torch.utils.data.SubsetRandomSampler(valid_idx))

        model = CheXNet(class_count=15, pretrained=False)
        checkpoint_callback = ModelCheckpoint(dirpath=f'checkpoints/nih-cxr/fold_{idx}',
                                              save_top_k=5, monitor='val_loss')
        trainer = pl.Trainer(max_epochs=50, accelerator='gpu', devices=1, callbacks=[checkpoint_callback])

        trainer.fit(model, dataloader_train, dataloader_valid)


if __name__ == '__main__':
    main()
