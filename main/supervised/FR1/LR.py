import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import random

# List of dataset filenames
dataset_files = ['D1_DATASET.xlsx',
                 'D2_DATASET.xlsx',
                 'D3_DATASET.xlsx',
                 'D4_DATASET.xlsx',
                 'D5_DATASET.xlsx',
                 'D6_DATASET.xlsx',
                 'D7_DATASET.xlsx',
                 'D8_DATASET.xlsx',
                 'D9_DATASET.xlsx',
                 'D10_DATASET.xlsx',
                 'D11_DATASET.xlsx',
                 'D12_DATASET.xlsx',
                 'D13_DATASET.xlsx',
                 'D14_DATASET.xlsx',
                 'D15_DATASET.xlsx',
                 'D16_DATASET.xlsx',
                 'D17_DATASET.xlsx',
                 'D18_DATASET.xlsx',
                 'D19_DATASET.xlsx',
                 'D20_DATASET.xlsx',
                 'D21_DATASET.xlsx',
                 'D22_DATASET.xlsx',
                 'D23_DATASET.xlsx',
                 'D24_DATASET.xlsx',
                 'D25_DATASET.xlsx',
                 'D26_DATASET.xlsx',
                 'D27_DATASET.xlsx',
                 'D28_DATASET.xlsx',
                 'D29_DATASET.xlsx',
                 'D30_DATASET.xlsx']

random.shuffle(dataset_files)

# Lists to hold accuracy and f1 score values for each dataset
accuracies = []
f_measures = []
confusion_matrices = []

for dataset_file in dataset_files:

    # Load dataset
    data = pd.read_excel(f'D:\\uit\\BaoMatWeb\\MLDroid\\DATASET\\{dataset_file}')

    # Shuffle the rows of the dataset
    data = data.sample(frac=1)

    # Perform one-hot encoding on the Package and Category columns
    data = pd.get_dummies(data, columns=['Package', 'Category'])

    # Replace missing values with the mean
    imputer = SimpleImputer(strategy='mean')
    X = data.drop('Class', axis=1)
    X = imputer.fit_transform(X)

    # Perform feature selection using gain ratio
    y = data['Class']
    selector = SelectKBest(chi2, k=20)
    X_new = selector.fit_transform(X, y)

    # Apply min-max normalization to the selected features
    scaler = MinMaxScaler()
    X_new = scaler.fit_transform(X_new)

    # Train a LR classifier using 20-fold cross-validation
    clf = LogisticRegression(random_state=16)
    sk_folds = StratifiedKFold(n_splits=20, shuffle=True ,random_state = None)
    
    for train_index, test_index in sk_folds.split(X_new, y):
        X_train, X_test = X_new[train_index], X_new[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Append confusion matrix to list of confusion matrices
        confusion_matrices.append(confusion_matrix(y_test, y_pred))

        # Append accuracy and f1 score to respective lists
        accuracy = clf.score(X_test, y_test)
        accuracies.append(accuracy)

        f_measure = f1_score(y_test, y_pred, average='weighted')
        f_measures.append(f_measure)

    # Calculate mean accuracy and f1 score
    accuracy_mean = np.mean(accuracies)
    f_measure_mean = np.mean(f_measures)

    # Print mean accuracy and f1 score
    print(f"{dataset_file}: Accuracy: {accuracy_mean*100:.2f}  F-measure: {f_measure_mean:.2f}")

#Result:
#D11_DATASET.xlsx: Accuracy: 88.19  F-measure: 0.84
#D23_DATASET.xlsx: Accuracy: 89.16  F-measure: 0.86
#D20_DATASET.xlsx: Accuracy: 89.03  F-measure: 0.86
#D8_DATASET.xlsx: Accuracy: 86.38  F-measure: 0.83
#D3_DATASET.xlsx: Accuracy: 86.35  F-measure: 0.83
#D16_DATASET.xlsx: Accuracy: 87.82  F-measure: 0.84
#D9_DATASET.xlsx: Accuracy: 87.79  F-measure: 0.84
#D17_DATASET.xlsx: Accuracy: 87.76  F-measure: 0.84
#D27_DATASET.xlsx: Accuracy: 86.97  F-measure: 0.83
#D12_DATASET.xlsx: Accuracy: 87.05  F-measure: 0.83
#D25_DATASET.xlsx: Accuracy: 87.46  F-measure: 0.84
#D2_DATASET.xlsx: Accuracy: 87.84  F-measure: 0.84
#D14_DATASET.xlsx: Accuracy: 88.14  F-measure: 0.85
#D18_DATASET.xlsx: Accuracy: 88.30  F-measure: 0.85
#D10_DATASET.xlsx: Accuracy: 88.46  F-measure: 0.85
#D19_DATASET.xlsx: Accuracy: 88.64  F-measure: 0.85
#D13_DATASET.xlsx: Accuracy: 89.05  F-measure: 0.86
#D7_DATASET.xlsx: Accuracy: 89.14  F-measure: 0.86
#D24_DATASET.xlsx: Accuracy: 88.84  F-measure: 0.85
#D15_DATASET.xlsx: Accuracy: 88.80  F-measure: 0.85
#D1_DATASET.xlsx: Accuracy: 88.43  F-measure: 0.85
#D29_DATASET.xlsx: Accuracy: 88.35  F-measure: 0.85
#D21_DATASET.xlsx: Accuracy: 87.95  F-measure: 0.84
#D5_DATASET.xlsx: Accuracy: 87.85  F-measure: 0.84
#D6_DATASET.xlsx: Accuracy: 87.86  F-measure: 0.84
#D22_DATASET.xlsx: Accuracy: 87.80  F-measure: 0.84
#D28_DATASET.xlsx: Accuracy: 87.84  F-measure: 0.84
#D4_DATASET.xlsx: Accuracy: 87.85  F-measure: 0.84
#D30_DATASET.xlsx: Accuracy: 87.91  F-measure: 0.84
#D26_DATASET.xlsx: Accuracy: 87.84  F-measure: 0.84