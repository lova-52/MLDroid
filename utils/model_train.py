from sklearn.metrics import f1_score, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB as BayesianClassifer
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import numpy as np
import random

def model_training(classifier_name, X_new, y, dataset_file):

    confusion_matrices = []
    accuracies = []
    f_measures = []
    
    if classifier_name == "SVM":
        # Train a SVM classifier using 20-fold cross-validation
        clf = SVC(kernel='poly', C=69, random_state=None, max_iter=1000)
        pass
    elif classifier_name == "NB":
        # Train a Naive Bayes classifier using 20-fold cross-validation
        clf = GaussianNB(var_smoothing=random.uniform(1e-9, 1e-6))
        pass
    elif classifier_name == "RF":
        # Train a Random Forest classifier using 20-fold cross-validation
        clf = RandomForestClassifier(n_estimators=100, random_state=None)
        pass
    elif classifier_name == "MLP":
        # Train a MLP classifier using 20-fold cross-validation
        clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=None)
        pass
    elif classifier_name == "LR":
        # Train a Logistic Regression classifier using 20-fold cross-validation
        clf = clf = LogisticRegression(random_state=None)
        pass
    elif classifier_name == "BN":
        # Train a Bayesian Network classifier using 20-fold cross-validation
        clf = BayesianClassifer()
        pass
    elif classifier_name == "AB":
        # Train an AdaBoost classifier using 20-fold cross-validation
        clf = AdaBoostClassifier(n_estimators=100, random_state=None)
        pass
    elif classifier_name == "DT":
        # Train a Decision Tree classifier using 20-fold cross-validation
        clf = DecisionTreeClassifier(random_state=None)
        pass
    elif classifier_name == "KNN":
        # Train a k-Nearest Neighbors classifier using 20-fold cross-validation
        clf = KNeighborsClassifier(weights=random.choice(['uniform', 'distance']))
        pass
    elif classifier_name == "DNN":
        # Train a Deep Neural Network classifier using 20-fold cross-validation
        clf = MLPClassifier(hidden_layer_sizes=(200, 150, 100, 50), max_iter=1000)
        pass
    else:
        print("Invalid classifier name.")

    sk_folds = StratifiedKFold(n_splits=20, shuffle=True, random_state=None)

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

    # Print mean accuracy and f1 score
    print(f"{dataset_file}: Accuracy: {accuracy_mean*100:.2f}  F-measure: {f_measure_mean:.2f}")

