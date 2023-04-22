import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib 
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def gain_ratio(X, y):
    n_features = X.shape[1]
    X = X.astype('int64')
    # Calculate information entropy of class variable
    class_counts = np.bincount(y)
    class_probs = class_counts / len(y)
    entropy = -np.sum(class_probs * np.log2(class_probs))

    # Calculate gain ratio for each feature
    gain_ratios = np.zeros(n_features)
    for i in range(n_features):
        feature_values = X[:, i]
        # Calculate information entropy of feature
        value_counts = np.bincount(feature_values)
        value_probs = value_counts / len(feature_values)
        value_entropy = -np.sum(value_probs * np.log2(value_probs))
        # Calculate split information of feature
        split_info = -np.sum(value_probs * np.log2(value_probs))
        # Calculate information gain of feature
        value_class_counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=len(class_counts)), 0, [feature_values, y])
        value_class_probs = value_class_counts / len(y)
        value_entropy_class = -np.sum(value_class_probs * np.log2(value_class_probs), axis=1)
        value_entropy_class_weighted = np.sum(value_probs * value_entropy_class)
        information_gain = entropy - value_entropy_class_weighted
        # Calculate gain ratio of feature
        if split_info == 0:
            gain_ratios[i] = 0
        else:
            gain_ratios[i] = information_gain / split_info
    return gain_ratios

# List of dataset filenames
dataset_files = [

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

# Lists to hold accuracy and f1 score values for each dataset
accuracies = []
f_measures = []

for dataset_file in dataset_files:

    # Load dataset
    data = pd.read_excel(f'D:\\uit\\BaoMatWeb\\MLDroid\\DATASET\\{dataset_file}')

    # Perform one-hot encoding on the Package and Category columns
    data = pd.get_dummies(data, columns=['Package', 'Category'])

    # Replace missing values with the mean
    imputer = SimpleImputer(strategy='mean')
    X = data.drop('Class', axis=1)
    X = imputer.fit_transform(X)

    # Perform feature selection using chi-squared test with k = k = log2(1400)~~11
    y = data['Class']
    selector = SelectKBest(gain_ratio, k=20)
    X_new = selector.fit_transform(X, y)

    # Apply min-max normalization to the selected features
    scaler = MinMaxScaler()
    X_new = scaler.fit_transform(X_new)

    # Train a DNN classifier using 20-fold cross-validation
    clf = MLPClassifier(hidden_layer_sizes=(200, 150, 100, 50), max_iter=1000)
    sk_folds = StratifiedKFold(n_splits = 20)
    clf.fit(X_new, y)
    scores = cross_val_score(clf, X_new, y, cv=sk_folds)
    accuracy = scores.mean()
    accuracies.append(accuracy)
    print(f"{dataset_file}: Accuracy: {accuracy:.4f}")

    # Make predictions on the full dataset and calculate f1 score
    clf.fit(X_new, y)
    f_measure = f1_score(y, clf.predict(X_new), average='weighted')
    f_measures.append(f_measure)
    print(f"{dataset_file}: F-measure: {f_measure:.4f}")

#    D20_DATASET.xlsx: Accuracy: 0.8553
#D20_DATASET.xlsx: F-measure: 0.7885
#D21_DATASET.xlsx: Accuracy: 0.7378
#D21_DATASET.xlsx: F-measure: 0.6265
#D22_DATASET.xlsx: Accuracy: 0.8264
#D22_DATASET.xlsx: F-measure: 0.7478
#D23_DATASET.xlsx: Accuracy: 0.8553
#D23_DATASET.xlsx: F-measure: 0.7886
#D24_DATASET.xlsx: Accuracy: 0.8188
#D24_DATASET.xlsx: F-measure: 0.7372
#D25_DATASET.xlsx: Accuracy: 0.9082
#D25_DATASET.xlsx: F-measure: 0.8646
#D26_DATASET.xlsx: Accuracy: 0.8314
#D26_DATASET.xlsx: F-measure: 0.7547
#D27_DATASET.xlsx: Accuracy: 0.7860
#D27_DATASET.xlsx: F-measure: 0.6918
#D28_DATASET.xlsx: Accuracy: 0.8882
#D28_DATASET.xlsx: F-measure: 0.8356
#D29_DATASET.xlsx: Accuracy: 0.8550
#D29_DATASET.xlsx: F-measure: 0.7882
#D30_DATASET.xlsx: Accuracy: 0.8275
#D30_DATASET.xlsx: F-measure: 0.7493
#Press any key to continue . . .