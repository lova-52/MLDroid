import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, f1_score
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
import joblib
import matplotlib.pyplot as plt

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

def select_features_Chi2(X, y, k):
    selector = SelectKBest(chi2, k=k)
    X_new = selector.fit_transform(X, y)
    return X_new


def apply_min_max_scaler(X):
    scaler = MinMaxScaler()
    X_new = scaler.fit_transform(X)
    return X_new


def train_classifiers(X, y):
    clf_ab = AdaBoostClassifier(n_estimators=50)
    clf_dnn = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
    clf_dt = DecisionTreeClassifier()
    clf_knn = KNeighborsClassifier()
    clf_lr = LogisticRegression()
    clf_mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
    clf_nb = GaussianNB()
    clf_rf = RandomForestClassifier(n_estimators=50)
    clf_svm = SVC(kernel='linear')

    classifiers = [clf_svm, clf_nb, clf_rf, clf_mlp, clf_lr, clf_ab, clf_dt, clf_dnn, clf_knn]

    return classifiers


def cross_validate(classifier, X, y, cv):
    scores = cross_val_score(classifier, X, y, cv=cv)
    accuracy = scores.mean()
    return accuracy


def fit_classifier(classifier, X, y):
    classifier.fit(X, y)
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

for dataset_file in dataset_files:

    # Load dataset
    data = load_dataset(f'D:\\uit\\BaoMatWeb\\MLDroid\\DATASET\\{dataset_file}')

    # Perform one-hot encoding on the Package and Category columns
    data = one_hot_encode(data, ['Package', 'Category'])

    # Replace missing values with the mean
    X = impute_mean(data)

    # Perform feature selection using chi-squared test with k = k = log2(1400)~~11
    y = data['Class']
    X_new = select_features_Chi2(X, y, 11)

    # Apply min-max normalization to the selected features
    X = apply_min_max_scaler(X)
    X_new = apply_min_max_scaler(X_new)

    # Train classifiers using 20-fold cross-validation
    classifiers = train_classifiers(X_new, y)

    for clf in classifiers:
        accuracy = cross_validate(clf, X_new, y, cv=20)
        accuracies.append(accuracy)
        print(f"{type(clf).__name__} ({dataset_file}): {accuracy}")

        f_measure = fit_classifier(clf, X_new, y)
        f_measures.append(f_measure)
        print(f"{type(clf).__name__} ({dataset_file}): {f_measure}")
