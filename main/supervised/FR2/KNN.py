import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
import warnings
from sklearn.metrics import f1_score, confusion_matrix
import random

#Define gain ratio function
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
print(dataset_files)

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
    selector = SelectKBest(gain_ratio, k=20)
    X_new = selector.fit_transform(X, y)

    # Apply min-max normalization to the selected features
    scaler = MinMaxScaler()
    X_new = scaler.fit_transform(X_new)

    # Train a Naive Bayes classifier using 20-fold cross-validation
    clf = clf_knn = KNeighborsClassifier(weights=random.choice(['uniform', 'distance']))
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
    print(f"{dataset_file}: Accuracy: {accuracy_mean:.4f}  F-measure: {f_measure_mean:.4f}")

    #Result:
    