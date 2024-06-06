import os
import cv2

import pandas as pd

import torch
from torch.utils.data import Dataset


class CheXpert(Dataset):

    def __init__(self, dataset_path, split='train', transform=None):

        self.data_root = os.path.join(dataset_path, '')
        self.dataset = pd.read_csv(os.path.join(dataset_path, f'{split}.csv'))
        self.dataset = self.dataset.fillna(0)
        self.dataset = self.dataset.replace(-1, 0)

        self.class_names = {
            0: "No Finding",
            1: "Enlg. Cardiomed.",
            2: "Cardiomegaly",  # common
            3: "Lung Opacity",  # common
            4: "Lung Lesion",  # common
            5: "Edema",
            6: "Consolidation",  # common
            7: "Pneumonia",
            8: "Atelectasis",  # common
            9: "Pneumothorax",
            10: "Pleural Effusion",  # common
            11: "Pleural Other",
            12: "Fracture",
            13: "Support Devices"
        }

        if split == 'train' or split == 'valid':
            self.listImagePaths = self.dataset['Path'].apply(lambda x: os.path.join(self.data_root, '..', x)).tolist()
            self.listImageLabels = self.dataset[self.dataset.columns[5:]].values.astype('int')
        elif split == 'test':
            self.listImagePaths = self.dataset['Path'].apply(lambda x: os.path.join(self.data_root, x)).tolist()
            self.listImageLabels = self.dataset[self.dataset.columns[1:]].values.astype('int')
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        image = cv2.imread(self.listImagePaths[idx])
        label = self.listImageLabels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class NIH_CXR(Dataset):

    def __init__(self, dataset_path, split='train_val', transform=None):

        self.data_root = os.path.join(dataset_path, '')

        self.class_names = {
            0: 'No Finding',
            1: 'Atelectasis',  # common
            2: 'Cardiomegaly',  # common
            3: 'Effusion',  # common
            4: 'Infiltration',
            5: 'Mass',  # common (opacity)
            6: 'Nodule',  # common
            7: 'Pneumonia',
            8: 'Pneumothorax',
            9: 'Consolidation',  # common
            10: 'Edema',
            11: 'Emphysema',
            12: 'Fibrosis',
            13: 'Pleural_Thickening',
            14: 'Hernia',
        }

        if split == 'train_val':
            self.dataset = pd.read_csv(os.path.join(dataset_path, 'train_val_list.csv'))
        elif split == 'test':
            self.dataset = pd.read_csv(os.path.join(dataset_path, 'test_list.csv'))

        self.listImagePaths = self.dataset['Image Index'].apply(lambda x: os.path.join(self.data_root, 'images', x)).tolist()
        self.listImageLabels = self.dataset[self.dataset.columns[10:]].values.astype('int')

        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        image = cv2.imread(self.listImagePaths[idx])
        label = self.listImageLabels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class DRR_RATE(Dataset):

    def __init__(self, dataset_path, split='train', transform=None):

        self.data_root = os.path.join(dataset_path, '')

        self.class_names = {
            0: "Cardiomegaly",  # common
            1: "Atelectasis",  # common
            2: "Lung Nodule",  # common
            3: "Lung Opacity",  # common
            4: "Pleural Effusion",  # common
            5: "Consolidation",  # common
        }

        self.dataset = pd.read_csv(os.path.join(dataset_path, f'multi_abnormality_labels/{split}_predicted_labels.csv'))

        self.listImagePaths = self.dataset['ImageName'].apply(
            lambda x: os.path.join(self.data_root, f'{split}/AP', self.get_path(x))).tolist()
        self.listImageLabels = self.dataset[[
            'Cardiomegaly', 'Atelectasis', 'Lung nodule',
            'Lung opacity', 'Pleural effusion', 'Consolidation']].values.astype('int')

        self.transform = transform

    def get_path(self, name):
        parts = name.split('_')
        folder = '_'.join(parts[:2])
        subfolder = '_'.join(parts[:3])
        rpath = os.path.join(folder, subfolder, name)
        return rpath


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        image = cv2.imread(self.listImagePaths[idx])
        label = self.listImageLabels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label





if __name__ == '__main__':

    dataloader = DRR_RATE('/vol/biomedic/users/bh1511/DRR-RATE')
    dataloader.__getitem__(12)

    print('Hello, World!')
