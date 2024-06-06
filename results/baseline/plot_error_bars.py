import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from glob import glob


# files = sorted(glob('nih-cxr/*_results.csv'))
files = sorted(glob('drr-rate/*_results.csv'))

results_df = [pd.read_csv(file, index_col=0) for file in files]
results_df = pd.concat(results_df, axis=1)

means1 = results_df['ROC_AUC'].mean(axis=1)
std_devs1 = results_df['ROC_AUC'].std(axis=1)

means2 = results_df['AP'].mean(axis=1)
std_devs2 = results_df['AP'].std(axis=1)

# Define class labels
classes = results_df.index.tolist()
class_indices = np.arange(len(classes))  # Numeric indices for class labels

# Create a figure and a set of subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 6))  # 1 row, 2 columns

# First subplot
ax1.errorbar(means1, class_indices, xerr=std_devs1, fmt='o', color='black', ecolor='red', capsize=5, linestyle='None') #, markersize=10)
ax1.set_title('ROC_AUC')
ax1.set_ylabel('Classes')
# ax1.set_xlabel('Predictions')
ax1.set_yticks(class_indices)
ax1.set_yticklabels(classes)
ax1.set_xlim(0.0, 1.0)
ax1.grid(True, which='major', axis='x', linestyle='-', linewidth='0.5', color='gray')

# Second subplot
ax2.errorbar(means2, class_indices, xerr=std_devs2, fmt='o', color='blue', ecolor='green', capsize=5, linestyle='None') #, markersize=10)
ax2.set_title('Average Precision')
# ax2.set_xlabel('Predictions')
ax2.set_yticks(class_indices)
ax2.set_yticklabels('')
ax2.set_xlim(0.0, 1.0)
ax2.grid(True, which='major', axis='x', linestyle='-', linewidth='0.5', color='gray')

# Improve layout
plt.tight_layout()

# Show the plot
plt.show()
