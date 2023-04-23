import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import BernoulliNB as BayesianClassifer
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

#Shuffle dataset files
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

    # Train a Bayesian classifier using 20-fold cross-validation
    clf = BayesianClassifer()
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
    #D12_DATASET.xlsx: Accuracy: 87.81  F-measure: 0.84
    #D6_DATASET.xlsx: Accuracy: 89.13  F-measure: 0.86
    #D1_DATASET.xlsx: Accuracy: 86.64  F-measure: 0.83
    #D4_DATASET.xlsx: Accuracy: 87.10  F-measure: 0.84
    #D2_DATASET.xlsx: Accuracy: 88.24  F-measure: 0.86
    #D18_DATASET.xlsx: Accuracy: 89.03  F-measure: 0.87
    #D22_DATASET.xlsx: Accuracy: 89.20  F-measure: 0.87
    #D23_DATASET.xlsx: Accuracy: 89.04  F-measure: 0.87
    #D28_DATASET.xlsx: Accuracy: 89.51  F-measure: 0.87
    #D24_DATASET.xlsx: Accuracy: 88.45  F-measure: 0.86
    #D17_DATASET.xlsx: Accuracy: 88.18  F-measure: 0.86
    #D26_DATASET.xlsx: Accuracy: 88.49  F-measure: 0.87
    #D13_DATASET.xlsx: Accuracy: 89.04  F-measure: 0.87
    #D10_DATASET.xlsx: Accuracy: 89.07  F-measure: 0.87
    #D16_DATASET.xlsx: Accuracy: 89.52  F-measure: 0.88
    #D15_DATASET.xlsx: Accuracy: 89.48  F-measure: 0.88
    #D7_DATASET.xlsx: Accuracy: 89.68  F-measure: 0.88
    #D30_DATASET.xlsx: Accuracy: 89.74  F-measure: 0.88
    #D11_DATASET.xlsx: Accuracy: 89.82  F-measure: 0.88
    #D20_DATASET.xlsx: Accuracy: 89.89  F-measure: 0.88
    #D29_DATASET.xlsx: Accuracy: 89.65  F-measure: 0.88
    #D25_DATASET.xlsx: Accuracy: 89.91  F-measure: 0.88
    #D19_DATASET.xlsx: Accuracy: 89.99  F-measure: 0.88
    #D14_DATASET.xlsx: Accuracy: 90.15  F-measure: 0.88
    #D5_DATASET.xlsx: Accuracy: 89.89  F-measure: 0.88
    #D3_DATASET.xlsx: Accuracy: 89.77  F-measure: 0.88
    #D9_DATASET.xlsx: Accuracy: 89.81  F-measure: 0.88
    #D21_DATASET.xlsx: Accuracy: 89.30  F-measure: 0.87
    #D8_DATASET.xlsx: Accuracy: 88.90  F-measure: 0.87
    #D27_DATASET.xlsx: Accuracy: 88.52  F-measure: 0.87. 