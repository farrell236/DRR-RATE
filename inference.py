import torch

import torchvision
from torchvision import transforms

import numpy as np
import pandas as pd

from cxr_dataset import CheXpert
from models import CheXNet

from tqdm import tqdm


# class_names = {
#     0: "No Finding",
#     1: "Enlg. Cardiomed.",
#     2: "Cardiomegaly",  # common
#     3: "Lung Opacity",  # common
#     4: "Lung Lesion",  # common
#     5: "Edema",
#     6: "Consolidation",  # common
#     7: "Pneumonia",
#     8: "Atelectasis",  # common
#     9: "Pneumothorax",
#     10: "Pleural Effusion",  # common
#     11: "Pleural Other",
#     12: "Fracture",
#     13: "Support Devices"
# }

# class_names = {
#     0: 'No Finding',
#     1: 'Atelectasis',  # common
#     2: 'Cardiomegaly',  # common
#     3: 'Effusion',  # common
#     4: 'Infiltration',
#     5: 'Mass',  # common (opacity)
#     6: 'Nodule',  # common
#     7: 'Pneumonia',
#     8: 'Pneumothorax',
#     9: 'Consolidation',  # common
#     10: 'Edema',
#     11: 'Emphysema',
#     12: 'Fibrosis',
#     13: 'Pleural_Thickening',
#     14: 'Hernia',
# }

class_names = {
    0: "Cardiomegaly",  # common
    1: "Atelectasis",  # common
    2: "Lung Nodule",  # common
    3: "Lung Opacity",  # common
    4: "Pleural Effusion",  # common
    5: "Consolidation",  # common
}


def main():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(512, scale=(1.0, 1.0), ratio=(1., 1.)),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    data_root = '/vol/biodata/data/CheXpert/CheXpert-v1.0'
    dataset_test = CheXpert(data_root, split='test', transform=transform)

    model = CheXNet(class_count=6, pretrained=False)
    # model.load_state_dict(torch.load('checkpoints/chexpert/epoch=22-step=160586.ckpt')['state_dict'])
    # model.load_state_dict(torch.load('checkpoints/nih-cxr/fold_0/epoch=19-step=43260.ckpt')['state_dict'])
    # model.load_state_dict(torch.load('checkpoints/nih-cxr/fold_1/epoch=21-step=47608.ckpt')['state_dict'])
    # model.load_state_dict(torch.load('checkpoints/nih-cxr/fold_2/epoch=21-step=47608.ckpt')['state_dict'])
    # model.load_state_dict(torch.load('checkpoints/nih-cxr/fold_3/epoch=21-step=47608.ckpt')['state_dict'])
    # model.load_state_dict(torch.load('checkpoints/nih-cxr/fold_4/epoch=18-step=41116.ckpt')['state_dict'])
    model.load_state_dict(torch.load('checkpoints/drr_rate/fold_0/epoch=16-step=20043.ckpt')['state_dict'])
    # model.load_state_dict(torch.load('checkpoints/drr_rate/fold_1/epoch=15-step=18864.ckpt')['state_dict'])
    # model.load_state_dict(torch.load('checkpoints/drr_rate/fold_2/epoch=16-step=20043.ckpt')['state_dict'])
    # model.load_state_dict(torch.load('checkpoints/drr_rate/fold_3/epoch=17-step=21222.ckpt')['state_dict'])
    # model.load_state_dict(torch.load('checkpoints/drr_rate/fold_4/epoch=17-step=21222.ckpt')['state_dict'])
    model.cuda()
    model.eval()

    # Iterate through the test dataset; store predictions and labels
    pred_df = pd.DataFrame(columns=list(class_names.values()))
    for idx, (image, _) in enumerate(tqdm(dataset_test)):
        y_pred = (model(image[None, ...].cuda())).cpu().detach().numpy()[0]
        pred_df.loc[idx] = y_pred

    pred_df.to_csv('results/drr-rate/chexpert-test-DRR-RATE-fold0.csv', index=False)

    a=1


if __name__ == '__main__':
    main()
