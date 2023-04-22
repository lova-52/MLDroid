import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
import numpy as np
import warnings
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

#Ignore warnings
warnings.filterwarnings("ignore")

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

    # Train a Naive Bayes classifier using 20-fold cross-validation
    clf = GaussianNB(var_smoothing=random.uniform(1e-9, 1e-6))
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
#D24_DATASET.xlsx: Mean Accuracy: 0.7885
#D24_DATASET.xlsx: Mean F-measure: 0.7882
#D14_DATASET.xlsx: Mean Accuracy: 0.8631
#D14_DATASET.xlsx: Mean F-measure: 0.8561
#D6_DATASET.xlsx: Mean Accuracy: 0.8769
#D6_DATASET.xlsx: Mean F-measure: 0.8637
#D18_DATASET.xlsx: Mean Accuracy: 0.8920
#D18_DATASET.xlsx: Mean F-measure: 0.8793
#D12_DATASET.xlsx: Mean Accuracy: 0.8892
#D12_DATASET.xlsx: Mean F-measure: 0.8715
#D17_DATASET.xlsx: Mean Accuracy: 0.8872
#D17_DATASET.xlsx: Mean F-measure: 0.8668
#D26_DATASET.xlsx: Mean Accuracy: 0.8918
#D26_DATASET.xlsx: Mean F-measure: 0.8723
#D4_DATASET.xlsx: Mean Accuracy: 0.8914
#D4_DATASET.xlsx: Mean F-measure: 0.8719
#D10_DATASET.xlsx: Mean Accuracy: 0.8935
#D10_DATASET.xlsx: Mean F-measure: 0.8730
#D16_DATASET.xlsx: Mean Accuracy: 0.9021
#D16_DATASET.xlsx: Mean F-measure: 0.8829
#D29_DATASET.xlsx: Mean Accuracy: 0.8967
#D29_DATASET.xlsx: Mean F-measure: 0.8775
#D3_DATASET.xlsx: Mean Accuracy: 0.8946
#D3_DATASET.xlsx: Mean F-measure: 0.8741
#D11_DATASET.xlsx: Mean Accuracy: 0.8963
#D11_DATASET.xlsx: Mean F-measure: 0.8761
#D23_DATASET.xlsx: Mean Accuracy: 0.8975
#D23_DATASET.xlsx: Mean F-measure: 0.8774
#D7_DATASET.xlsx: Mean Accuracy: 0.9004
#D7_DATASET.xlsx: Mean F-measure: 0.8806
#D8_DATASET.xlsx: Mean Accuracy: 0.8927
#D8_DATASET.xlsx: Mean F-measure: 0.8726
#D28_DATASET.xlsx: Mean Accuracy: 0.8962
#D28_DATASET.xlsx: Mean F-measure: 0.8767
#D5_DATASET.xlsx: Mean Accuracy: 0.8949
#D5_DATASET.xlsx: Mean F-measure: 0.8750
#D13_DATASET.xlsx: Mean Accuracy: 0.8994
#D13_DATASET.xlsx: Mean F-measure: 0.8803
#D20_DATASET.xlsx: Mean Accuracy: 0.9006
#D20_DATASET.xlsx: Mean F-measure: 0.8816
#D1_DATASET.xlsx: Mean Accuracy: 0.8957
#D1_DATASET.xlsx: Mean F-measure: 0.8751
#D25_DATASET.xlsx: Mean Accuracy: 0.8983
#D25_DATASET.xlsx: Mean F-measure: 0.8782
#D15_DATASET.xlsx: Mean Accuracy: 0.8982
#D15_DATASET.xlsx: Mean F-measure: 0.8780
#D9_DATASET.xlsx: Mean Accuracy: 0.8986
#D9_DATASET.xlsx: Mean F-measure: 0.8783
#D27_DATASET.xlsx: Mean Accuracy: 0.8938
#D27_DATASET.xlsx: Mean F-measure: 0.8732
#D30_DATASET.xlsx: Mean Accuracy: 0.8947
#D30_DATASET.xlsx: Mean F-measure: 0.8744
#D22_DATASET.xlsx: Mean Accuracy: 0.8950
#D22_DATASET.xlsx: Mean F-measure: 0.8747
#D2_DATASET.xlsx: Mean Accuracy: 0.8964
#D2_DATASET.xlsx: Mean F-measure: 0.8763
#D19_DATASET.xlsx: Mean Accuracy: 0.8973
#D19_DATASET.xlsx: Mean F-measure: 0.8771
#D21_DATASET.xlsx: Mean Accuracy: 0.8928
#D21_DATASET.xlsx: Mean F-measure: 0.8712