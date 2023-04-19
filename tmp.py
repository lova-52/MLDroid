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
    data = pd.read_excel(f'D:\\uit\\BaoMatWeb\\MLDroid\\DATASET\\{dataset_file}')

    # Perform one-hot encoding on the Package and Category columns
    data = pd.get_dummies(data, columns=['Package', 'Category'])

    # Replace missing values with the mean
    imputer = SimpleImputer(strategy='mean')
    X = data.drop('Class', axis=1)
    X = imputer.fit_transform(X)

    # Perform feature selection using chi-squared test with k = k = log2(1400)~~11
    y = data['Class']
    selector = SelectKBest(chi2, k=11)
    X_new = selector.fit_transform(X, y)

    # Apply min-max normalization to the selected features
    scaler = MinMaxScaler()
    X_new = scaler.fit_transform(X_new)

    # Train classifiers using 20-fold cross-validation
    clf_ab = AdaBoostClassifier(n_estimators=50)
    clf_dnn = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
    clf_dt = DecisionTreeClassifier()
    clf_knn = KNeighborsClassifier()
    clf_lr = LogisticRegression()
    clf_mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
    clf_nb = GaussianNB()
    clf_rf = RandomForestClassifier(n_estimators=50)
    clf_svm = SVC(kernel='linear')


    classifiers = [clf_svm, clf_nb, clf_rf, clf_mlp, clf_lr, clf_ab, clf_dt, clf_dnn,  clf_knn]

    for clf in classifiers:
        scores = cross_val_score(clf, X_new, y, cv=20)
        accuracy = scores.mean()
        accuracies.append(accuracy)
        print(f"{type(clf).__name__} ({dataset_file}): {accuracy}")

        # Make predictions on the full dataset and calculate f1 score
        clf.fit(X, y)
        f_measure = f1_score(y, clf.predict(X_new), average='weighted')
        f_measures.append(f_measure)
        print(f"{type(clf).__name__} ({dataset_file}): {f_measure}")
