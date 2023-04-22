import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import StratifiedKFold
import joblib
import numpy as np
import matplotlib.pyplot as plt
import random as random
import warnings
from sklearn.metrics import f1_score, confusion_matrix
warnings.filterwarnings("ignore")

frac = 1

def load_dataset(filename):
    data = pd.read_excel(filename)
    return data

def one_hot_encode(data, columns):
    data = pd.get_dummies(data, columns=columns)
    return data

def impute_mean(data):
    imputer = SimpleImputer(strategy='mean')
    X = data.drop('Class', axis=1)
    X = imputer.fit_transform(X)
    return X

# Perform feature selection using chi square
def select_features_chi2(X, y, k):
    selector = SelectKBest(chi2, k=k)
    X_new = selector.fit_transform(X, y)
    return X_new


def apply_min_max_scaler(X):
    scaler = MinMaxScaler()
    X_new = scaler.fit_transform(X)
    return X_new


def train_classifiers(X, y):
    clf_ab = AdaBoostClassifier(n_estimators=1000, learning_rate=random.uniform(0.01, 1), random_state=62)
    clf_dnn = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=64)
    clf_dt = DecisionTreeClassifier(max_depth=random.randint(20, 30), random_state=50)
    clf_knn = KNeighborsClassifier(weights=random.choice(['uniform', 'distance']))
    clf_lr = LogisticRegression(C=random.uniform(0.01, 100), random_state=23)
    clf_mlp = MLPClassifier(hidden_layer_sizes=(100, 50, 50, 50, 50), max_iter=1000, alpha=random.uniform(0.0001, 1), random_state=42)
    clf_nb = GaussianNB(var_smoothing=random.uniform(1e-9, 1e-6))
    clf_rf = RandomForestClassifier(n_estimators=1000, random_state=442)
    clf_svm = SVC(kernel='poly', C=69, random_state=None, max_iter=1000)

    classifiers = [clf_svm, clf_nb, clf_rf, clf_mlp, clf_lr, clf_ab, clf_dt, clf_knn, clf_dnn]

    return classifiers


def cross_validate(classifier, X, y, cv):
    scores = cross_val_score(classifier, X, y, cv=cv)
    accuracy = scores.mean()
    return accuracy


def fit_classifier(classifier, X, y):
    classifier.fit(X, y)
    #F1_score = (2 x precision_weighted x recall_weighted) / (precision_weighted + recall_weighted)
    f_measure = f1_score(y, classifier.predict(X), average='weighted') 
    return f_measure


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

# Lists to hold accuracy and f1 score values for each dataset
accuracies = []
f_measures = []
confusion_matrices = []

for dataset_file in dataset_files:

    # Load dataset
    data = load_dataset(f'D:\\uit\\BaoMatWeb\\MLDroid\\DATASET\\{dataset_file}')

    # Shuffle the rows of the dataset
    data = data.sample(frac)
    frac += 1

    # Perform one-hot encoding on the Package and Category columns
    data = one_hot_encode(data, ['Package', 'Category'])

    # Replace missing values with the mean
    X = impute_mean(data)

    # Perform feature selection using chi-squared test with k = 20
    y = data['Class']
    X_new = select_features_chi2(X, y, 20)

    # Apply min-max normalization to the selected features
    X_new = apply_min_max_scaler(X_new)

    # Train classifiers using 20-fold cross-validation
    classifiers = train_classifiers(X_new, y)
    sk_folds = StratifiedKFold(n_splits=20, shuffle=True ,random_state = None)

    for clf in classifiers:
        for train_index, test_index in sk_folds.split(X_new, y):
            X_train, X_test = X_new[train_index], X_new[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
    
            confusion_matrices.append(confusion_matrix(y_test, y_pred))
    
            accuracy = clf.score(X_test, y_test)
            accuracies.append(accuracy)
    
            f_measure = f1_score(y_test, y_pred, average='weighted')
            f_measures.append(f_measure)

        accuracy_mean = np.mean(accuracies)
        f_measure_mean = np.mean(f_measures)
    
        print(f"{dataset_file}: Mean Accuracy: {accuracy_mean:.4f}")        
        print(f"{dataset_file}: Mean F-measure: {f_measure_mean:.4f}")

    print(f"\\n")
