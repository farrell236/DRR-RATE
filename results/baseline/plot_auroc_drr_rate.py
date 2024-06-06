import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import *

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np




y_pred_0 = pd.read_csv('drr-rate/fold_0_y_pred.csv')
y_pred_1 = pd.read_csv('drr-rate/fold_1_y_pred.csv')
y_pred_2 = pd.read_csv('drr-rate/fold_2_y_pred.csv')
y_pred_3 = pd.read_csv('drr-rate/fold_3_y_pred.csv')
y_pred_4 = pd.read_csv('drr-rate/fold_4_y_pred.csv')
dfs = [y_pred_0, y_pred_1, y_pred_2, y_pred_3, y_pred_4]
y_pred = pd.concat(dfs).groupby(level=0).mean()
y_true = pd.read_csv('drr-rate/fold_0_y_true.csv')


a=1


stacked_dfs = pd.concat(dfs, axis=0)
# t = y_true[['Cardiomegaly', 'Atelectasis', 'Lung Lesion', 'Lung Opacity', 'Pleural Effusion', 'Consolidation']]
for i in range(6):

    # Compute mean and standard deviation for predictions
    average_df = stacked_dfs.groupby(level=0).mean().iloc[:, i]
    std_deviation = stacked_dfs.groupby(level=0).std().iloc[:, i]

    # True labels
    true_labels = y_true.iloc[:, i]  # Your true labels here

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(true_labels, average_df)
    roc_auc = auc(fpr, tpr)

    # Calculate the standard deviations for each threshold
    std_dev_at_thresholds = [std_deviation.iloc[(average_df <= thresh).values].mean() for thresh in thresholds]

    # Increase font size globally
    plt.rcParams.update({'font.size': 28})

    # Plot ROC Curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', color='b')

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

