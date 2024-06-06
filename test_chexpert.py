import torch

import torchvision
from torchvision import transforms

import numpy as np
import pandas as pd

from cxr_dataset import CheXpert
from models import CheXNet

from sklearn.metrics import *
from tqdm import tqdm


def main():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(512, scale=(1.0, 1.0), ratio=(1., 1.)),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    data_root = '/vol/biodata/data/CheXpert/CheXpert-v1.0'
    dataset_test = CheXpert(data_root, split='test', transform=transform)

    model = CheXNet(class_count=14, pretrained=False)
    model.load_state_dict(torch.load('checkpoints/chexpert/epoch=22-step=160586.ckpt')['state_dict'])
    model.cuda()
    model.eval()

    # Iterate through the test dataset; store predictions and labels
    y_pred_df = pd.DataFrame(columns=list(dataset_test.class_names.values()))
    y_true_df = pd.DataFrame(columns=list(dataset_test.class_names.values()))
    for idx, (image, label) in enumerate(tqdm(dataset_test)):
        y_pred = (model(image[None, ...].cuda())).cpu().detach().numpy()[0]
        y_pred_df.loc[idx] = y_pred
        y_true_df.loc[idx] = label

    # Calculate ROC AUC and AP scores for each class
    results_df = pd.DataFrame(columns=['ROC_AUC', 'AP'])
    for c in dataset_test.class_names.values():
        results_df.loc[c] = [
            roc_auc_score(y_true_df[c], y_pred_df[c]),
            average_precision_score(y_true_df[c], y_pred_df[c])]

    print(results_df)

    # Write results to CSV
    y_pred_df.to_csv('results/baseline/chexpert/y_pred.csv', index=False)
    y_true_df.to_csv('results/baseline/chexpert/y_true.csv', index=False)
    results_df.to_csv('results/baseline/chexpert/results.csv')


    a=1


if __name__ == '__main__':
    main()
