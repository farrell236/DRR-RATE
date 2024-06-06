import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import *


y_pred = pd.read_csv('chexpert/y_pred.csv')
y_true = pd.read_csv('chexpert/y_true.csv')

# y_pred = pd.read_csv('drr-rate/fold_0_y_pred.csv')
# y_true = pd.read_csv('drr-rate/fold_0_y_true.csv')

a=1

# roc_auc_score(y_true['Atelectasis'], y_pred['Atelectasis'])
# roc_auc_score(y_true['Cardiomegaly'], y_pred['Cardiomegaly'])
# roc_auc_score(y_true['Pleural effusion'], y_pred['Effusion'])
# roc_auc_score(y_true['Consolidation'], y_pred['Consolidation'])
# roc_auc_score(y_true['Lung nodule'], y_pred['Nodule'])
# roc_auc_score(y_true['Lung opacity'], y_pred['Mass'])
# average_precision_score(y_true['Atelectasis'], y_pred['Atelectasis'])
# average_precision_score(y_true['Cardiomegaly'], y_pred['Cardiomegaly'])
# average_precision_score(y_true['Pleural effusion'], y_pred['Effusion'])
# average_precision_score(y_true['Consolidation'], y_pred['Consolidation'])
# average_precision_score(y_true['Lung nodule'], y_pred['Nodule'])
# average_precision_score(y_true['Lung opacity'], y_pred['Mass'])

def plot_auroc(y_true, y_pred, name):

    fpr, tpr, threshold = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # Increase font size globally
    plt.rcParams.update({'font.size': 20})

    plt.title(f'ROC: {name}')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.tight_layout()
    plt.savefig(f'{name}.pdf')
    plt.show()




plot_auroc(y_true['Atelectasis'], y_pred['Atelectasis'], 'Atelectasis')
plot_auroc(y_true['Cardiomegaly'], y_pred['Cardiomegaly'], 'Cardiomegaly')
plot_auroc(y_true['Pleural Effusion'], y_pred['Pleural Effusion'], 'Pleural Effusion')
plot_auroc(y_true['Consolidation'], y_pred['Consolidation'], 'Consolidation')
plot_auroc(y_true['Lung Lesion'], y_pred['Lung Lesion'], 'Lung Lesion')
plot_auroc(y_true['Lung Opacity'], y_pred['Lung Opacity'], 'Lung Opacity')



# plot_auroc(y_true['Atelectasis'], y_pred['Atelectasis'], 'Atelectasis')
# plot_auroc(y_true['Cardiomegaly'], y_pred['Cardiomegaly'], 'Cardiomegaly')
# plot_auroc(y_true['Pleural Effusion'], y_pred['Pleural Effusion'], 'Pleural Effusion')
# plot_auroc(y_true['Consolidation'], y_pred['Consolidation'], 'Consolidation')
# plot_auroc(y_true['Lung Nodule'], y_pred['Lung Nodule'], 'Lung Nodule')
# plot_auroc(y_true['Lung Opacity'], y_pred['Lung Opacity'], 'Lung Opacity')


a=1
