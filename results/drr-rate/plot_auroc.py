import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import *

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np

y_pred_0 = pd.read_csv('chexpert-test-DRR-RATE-fold0.csv')
y_pred_1 = pd.read_csv('chexpert-test-DRR-RATE-fold1.csv')
y_pred_2 = pd.read_csv('chexpert-test-DRR-RATE-fold2.csv')
y_pred_3 = pd.read_csv('chexpert-test-DRR-RATE-fold3.csv')
y_pred_4 = pd.read_csv('chexpert-test-DRR-RATE-fold4.csv')
dfs = [y_pred_0, y_pred_1, y_pred_2, y_pred_3, y_pred_4]
y_pred = pd.concat(dfs).groupby(level=0).mean()
y_true = pd.read_csv('/vol/biodata/data/CheXpert/CheXpert-v1.0/test.csv')


a=1


stacked_dfs = pd.concat(dfs, axis=0)
t = y_true[['Cardiomegaly', 'Atelectasis', 'Lung Lesion', 'Lung Opacity', 'Pleural Effusion', 'Consolidation']]
for i in range(6):

    # Compute mean and standard deviation for predictions
    average_df = stacked_dfs.groupby(level=0).mean().iloc[:, i]
    std_deviation = stacked_dfs.groupby(level=0).std().iloc[:, i]

    # True labels
    true_labels = t.iloc[:, i]  # Your true labels here

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(true_labels, average_df)

    # Calculate the standard deviations for each threshold
    std_dev_at_thresholds = [std_deviation.iloc[(average_df <= thresh).values].mean() for thresh in thresholds]

    # Increase font size globally
    plt.rcParams.update({'font.size': 24})

    # Plot ROC Curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label='ROC Curve', color='b')

    # Add shaded area for uncertainty
    tpr_upper = np.clip(tpr + np.array(std_dev_at_thresholds), 0, 1)
    tpr_lower = np.clip(tpr - np.array(std_dev_at_thresholds), 0, 1)
    plt.fill_between(fpr, tpr_lower, tpr_upper, color='grey', alpha=0.3, label='Uncertainty')

    # Add details
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {y_pred.columns[i]}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f'{y_pred.columns[i]}.pdf')
    plt.show()














def plot_auroc(y_true, y_pred, name):

    fpr, tpr, threshold = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # Increase font size globally
    plt.rcParams.update({'font.size': 12})

    plt.title(f'Receiver Operating Characteristic: {name}')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(f'{name}.pdf')
    plt.show()




plot_auroc(y_true['Atelectasis'], y_pred['Atelectasis'], 'Atelectasis')
plot_auroc(y_true['Cardiomegaly'], y_pred['Cardiomegaly'], 'Cardiomegaly')
plot_auroc(y_true['Pleural Effusion'], y_pred['Pleural Effusion'], 'Pleural Effusion')
plot_auroc(y_true['Consolidation'], y_pred['Consolidation'], 'Consolidation')
plot_auroc(y_true['Lung Lesion'], y_pred['Lung Nodule'], 'Lung Nodule')
plot_auroc(y_true['Lung Opacity'], y_pred['Lung Opacity'], 'Lung Opacity')


