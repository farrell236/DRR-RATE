import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


metadata_train = pd.read_csv('/vol/biomedic/users/bh1511/DRR-RATE/metadata/train_metadata.csv')
metadata_valid = pd.read_csv('/vol/biomedic/users/bh1511/DRR-RATE/metadata/validation_metadata.csv')


# Function to convert age
def convert_age(age_str):
    if pd.isna(age_str):
        return np.nan
    elif 'Y' in age_str:
        return int(age_str.replace('Y', ''))
    elif 'M' in age_str:
        return int(age_str.replace('M', '')) / 12
    else:
        return np.nan

metadata_train['PatientID'] = metadata_train['ImageName'].str.extract(r'train_(\d+)_')
metadata_train['PatientAge'] = metadata_train['PatientAge'].apply(convert_age)

metadata_valid['PatientID'] = metadata_valid['ImageName'].str.extract(r'valid_(\d+)_')
metadata_valid['PatientAge'] = metadata_valid['PatientAge'].apply(convert_age)

unique_patients_train = metadata_train.drop_duplicates(subset='PatientID', keep='first')
unique_patients_valid = metadata_valid.drop_duplicates(subset='PatientID', keep='first')

# Drop NaN values for the histogram
age_data_train = unique_patients_train['PatientAge'].dropna()
age_data_valid = unique_patients_valid['PatientAge'].dropna()

# Increase font size globally
plt.rcParams.update({'font.size': 24})

# Plotting the histograms
fig, ax1 = plt.subplots(figsize=(12, 8))

# Histogram for metadata1
ax1.hist(age_data_train, bins=10, edgecolor='black', alpha=0.5, label='Training Set')
ax1.set_ylim(0, 5000)
ax1.set_xlabel('Age (years)')
ax1.set_ylabel('Training Frequency')
ax1.grid(True)

# Create a secondary y-axis for metadata2
ax2 = ax1.twinx()
ax2.hist(age_data_valid, bins=10, edgecolor='black', alpha=0.5, color='orange', label='Validation Set')
ax2.set_ylim(0, 500)
ax2.set_ylabel('Validation Frequency')

# Combine legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper right')

plt.title('Distribution of Patient Age')
plt.savefig('age_hist.pdf')
plt.show()

print(age_data_train.describe())
print(age_data_valid.describe())


a=1

m1, f1 = unique_patients_train['PatientSex'].value_counts().values
m2, f2 = unique_patients_valid['PatientSex'].value_counts()



# Sample data for gender distribution
# Replace these with your actual data
train_gender_distribution = {'Male': m1, 'Female': f1}
val_gender_distribution = {'Male': m2, 'Female': f2}

# Data for plotting
labels = list(train_gender_distribution.keys())
train_values = list(train_gender_distribution.values())
val_values = list(val_gender_distribution.values())

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax1 = plt.subplots(figsize=(10, 8))

# Plotting the training set bars
bars1 = ax1.bar(x - width/2, train_values, width, label='Training Set', edgecolor='black', alpha=0.5)
ax1.set_ylim(0, 12000)
ax1.set_xlabel('Gender')
ax1.set_ylabel('Training Frequency')
ax1.set_title('Gender Distribution')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.grid(True)

# Creating a secondary y-axis for the validation set
ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, val_values, width, label='Validation Set', color='orange', edgecolor='black', alpha=0.5)
ax2.set_ylim(0, 1200)
ax2.set_ylabel('Validation Frequency')

# Combine legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper right')


# Show the plot
plt.tight_layout()
plt.savefig('gender_hist.pdf')
plt.show()

a=1

train_labels = pd.read_csv('/vol/biomedic/users/bh1511/DRR-RATE/multi_abnormality_labels/train_predicted_labels.csv')
valid_labels = pd.read_csv('/vol/biomedic/users/bh1511/DRR-RATE/multi_abnormality_labels/valid_predicted_labels.csv')

# train_labels['PatientID'] = train_labels['ImageName'].str.extract(r'train_(\d+)_')
# valid_labels['PatientID'] = valid_labels['ImageName'].str.extract(r'valid_(\d+)_')
#
# unique_patients_train = train_labels.drop_duplicates(subset='PatientID', keep='first')
# unique_patients_valid = valid_labels.drop_duplicates(subset='PatientID', keep='first')
#
# train_count = unique_patients_train[unique_patients_train.columns[1:-1]].sum(axis=0)
# valid_count = unique_patients_valid[unique_patients_valid.columns[1:-1]].sum(axis=0)

train_count = train_labels[train_labels.columns[1:]].sum(axis=0)
valid_count = valid_labels[valid_labels.columns[1:]].sum(axis=0)

label_count = pd.concat([train_count, valid_count], axis=1)
label_count = label_count.rename(columns={0: 'Training Set', 1: 'Validation Set'})
label_count['Ratio'] = label_count['Validation Set'] / label_count['Training Set']

a=1

