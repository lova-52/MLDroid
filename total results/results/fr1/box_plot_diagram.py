import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data from Excel file
df = pd.read_excel(f'D:\\uit\\BaoMatWeb\\MLDroid\\DroidFusion\\main\\supervised\\FR1\\results\\accuracy_fmeasure_fr1.xlsx', skiprows=1)
print(df.columns)

# set the columns for accuracy and f-measure
acc_cols = ['SVM', 'NB', 'RF', 'MLP', 'LR', 'BN', 'AB', 'DT', 'KNN', 'DNN']
fme_cols = ['SVM.1', 'NB.1', 'RF.1', 'MLP.1', 'LR.1', 'BN.1', 'AB.1', 'DT.1', 'KNN.1', 'DNN.1']

# create separate plots for accuracy and f-measure box plots
fig1, ax1 = plt.subplots(figsize=(10, 5))
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax1.set_ylim([70, 100])
ax2.set_ylim([0.7, 1])

# plot accuracy box plot
sns.boxplot(data=df[acc_cols], ax=ax1)
ax1.set_title('Accuracy')

# plot f-measure box plot
sns.boxplot(data=df[fme_cols], ax=ax2)
ax2.set_title('F-Measure')

plt.show()